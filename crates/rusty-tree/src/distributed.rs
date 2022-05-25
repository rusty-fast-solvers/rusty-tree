//! Data structures and methods to create distributed Octrees with MPI.

use std::collections::{HashMap, HashSet};
use std::convert::TryInto;

use mpi::{
    datatype::{Partition, PartitionMut},
    topology::{Rank, UserCommunicator},
    traits::*,
    Count,
};

use hyksort::hyksort::hyksort;

use crate::{
    constants::{K, NCRIT, ROOT},
    data::{HDF5, JSON, VTK},
    single_node::Tree,
    types::{
        domain::Domain,
        morton::MortonKey,
        point::{Point, PointType},
    },
};

/// Interface for a distributed tree, adaptive by default.
pub struct DistributedTree {
    /// Balancing is optional.
    pub balanced: bool,

    ///  A vector of Cartesian points.
    pub points: Vec<Point>,

    /// Map between the point's minimum leaf encoding, and the nodes in the tree.
    pub points_to_keys: HashMap<MortonKey, MortonKey>,

    /// The nodes that span the tree, defined by its leaf nodes.
    pub keys: Vec<MortonKey>,

    /// Domain spanned by the points in the tree.
    pub domain: Domain,
}

impl DistributedTree {
    /// Create a new DistributedTree from a set of distributed points which define a domain.
    pub fn new(
        points: &[[PointType; 3]],
        balanced: bool,
        world: &UserCommunicator,
    ) -> DistributedTree {
        let domain = Domain::from_global_points(&points, world);

        if balanced {
            let (points, points_to_keys) = DistributedTree::balanced_tree(world, points, &domain);
            let keys = points.iter().map(|p| p.key).collect();

            DistributedTree {
                balanced,
                points,
                keys,
                points_to_keys,
                domain,
            }
        } else {
            let (points, points_to_keys) = DistributedTree::unbalanced_tree(world, points, &domain);
            let keys = points.iter().map(|p| p.key).collect();

            DistributedTree {
                balanced,
                points,
                keys,
                points_to_keys,
                domain,
            }
        }
    }

    /// Complete a distributed block tree from the seed octants, algorithm 4 in [1] (parallel).
    fn complete_blocktree(
        seeds: &mut Vec<MortonKey>,
        &rank: &Rank,
        &size: &Rank,
        world: &UserCommunicator,
    ) -> Tree {
        // Define the tree's global domain with the finest first/last descendants
        if rank == 0 {
            let ffc_root = ROOT.finest_first_child();
            let min = seeds.iter().min().unwrap();
            let fa = ffc_root.finest_ancestor(min);
            let first_child = fa.children().into_iter().min().unwrap();
            seeds.push(first_child);
            seeds.sort();
        }

        if rank == (size - 1) {
            let flc_root = ROOT.finest_last_child();
            let max = seeds.iter().max().unwrap();
            let fa = flc_root.finest_ancestor(max);
            let last_child = fa.children().into_iter().max().unwrap();
            seeds.push(last_child);
        }

        let next_rank = if rank + 1 < size { rank + 1 } else { 0 };
        let previous_rank = if rank > 0 { rank - 1 } else { size - 1 };

        let previous_process = world.process_at_rank(previous_rank);
        let next_process = world.process_at_rank(next_rank);

        // Send required data to partner process.
        if rank > 0 {
            let min = *seeds.iter().min().unwrap();
            previous_process.send(&min);
        }

        let mut boundary = MortonKey::default();

        if rank < (size - 1) {
            next_process.receive_into(&mut boundary);
            seeds.push(boundary);
        }

        // Complete region between seeds at each process
        let mut complete = Tree { keys: Vec::new() };

        for i in 0..(seeds.iter().len() - 1) {
            let a = seeds[i];
            let b = seeds[i + 1];

            let mut tmp: Vec<MortonKey> = Tree::complete_region(&a, &b);
            complete.keys.push(a);
            complete.keys.append(&mut tmp);
        }

        if rank == (size - 1) {
            complete.keys.push(*seeds.last().unwrap());
        }

        complete.sort();
        complete
    }

    /// Create a mapping between octree elements at a coarser (nodes) and finer (leaves) level.
    fn assign_nodes_to_leaves(
        leaves: &Vec<MortonKey>,
        nodes: &Vec<MortonKey>,
    ) -> HashMap<MortonKey, MortonKey> {
        let nodes: HashSet<MortonKey> = nodes.iter().cloned().collect();
        let mut map: HashMap<MortonKey, MortonKey> = HashMap::new();

        for leaf in leaves.iter() {
            if nodes.contains(leaf) {
                map.insert(*leaf, *leaf);
            } else {
                let mut ancestors: Vec<MortonKey> = leaf.ancestors().into_iter().collect();
                ancestors.sort();
                for ancestor in ancestors {
                    if nodes.contains(&ancestor) {
                        map.insert(*leaf, ancestor);
                        break;
                    }
                }
            };
        }
        map
    }

    /// Split octree nodes (blocks) by counting how many particles they contain.
    fn split_blocks(
        leaves: &Vec<MortonKey>,
        mut blocktree: Vec<MortonKey>,
    ) -> HashMap<MortonKey, MortonKey> {
        let mut split_blocktree: Vec<MortonKey> = Vec::new();

        loop {
            let mut new_blocktree: Vec<MortonKey> = Vec::new();

            // Map between blocks and the leaves they contain
            let blocks_to_leaves = DistributedTree::assign_nodes_to_leaves(leaves, &blocktree);

            // Count the number of points in a block
            let mut blocks_to_npoints: HashMap<MortonKey, usize> = HashMap::new();
            for (_, block) in blocks_to_leaves {
                if let std::collections::hash_map::Entry::Vacant(e) = blocks_to_npoints.entry(block)
                {
                    e.insert(1);
                } else if let Some(b) = blocks_to_npoints.get_mut(&block) {
                    *b += 1;
                }
            }

            // Generate a new blocktree with a block's children if they violate the NCRIT constraint
            let mut check = 0;
            for (&block, &npoints) in blocks_to_npoints.iter() {
                if npoints > NCRIT {
                    let mut children = block.children();
                    new_blocktree.append(&mut children);
                } else {
                    new_blocktree.push(block);
                    check += 1;
                }
            }

            if check == blocks_to_npoints.len() {
                split_blocktree = new_blocktree;
                break;
            } else {
                blocktree = new_blocktree;
            }
        }

        // Assign nodes in the split blocktree to all leaves
        DistributedTree::assign_nodes_to_leaves(leaves, &split_blocktree)
    }

    /// Find the seeds, defined as coarsest leaf/leaves, at each processor [1].
    fn find_seeds(leaves: &[MortonKey]) -> Vec<MortonKey> {
        let min: MortonKey = *leaves.iter().min().unwrap();
        let max: MortonKey = *leaves.iter().max().unwrap();

        // Complete the region between the least and greatest leaves.
        let mut complete = Tree::complete_region(&min, &max);
        complete.push(min);
        complete.push(max);

        // Find seeds by filtering for leaves at coarsest level
        let coarsest_level = complete.iter().map(|k| k.level()).min().unwrap();
        let mut seeds: Vec<MortonKey> = complete
            .into_iter()
            .filter(|k| k.level() == coarsest_level)
            .collect();

        seeds.sort();
        seeds
    }

    // Transfer points based on the coarse distributed blocktree.
    fn transfer_points_to_blocktree(
        world: &UserCommunicator,
        points: &[Point],
        seeds: &[MortonKey],
        &rank: &Rank,
        &size: &Rank,
    ) -> Vec<Point> {
        let mut received_points: Vec<Point> = Vec::new();

        let mut min_seed = MortonKey::default();

        if rank == 0 {
            min_seed = points.iter().min().unwrap().key;
        } else {
            min_seed = *seeds.iter().min().unwrap();
        }

        let prev_rank = if rank > 0 { rank - 1 } else { size - 1 };
        let next_rank = if rank + 1 < size { rank + 1 } else { 0 };

        if rank > 0 {
            let msg: Vec<Point> = points
                .iter()
                .filter(|&p| p.key < min_seed)
                .cloned()
                .collect();

            let msg_size: Rank = msg.len() as Rank;
            world.process_at_rank(prev_rank).send(&msg_size);
            world.process_at_rank(prev_rank).send(&msg[..]);
        }

        if rank < (size - 1) {
            let mut bufsize = 0;
            world.process_at_rank(next_rank).receive_into(&mut bufsize);
            let mut buffer = vec![Point::default(); bufsize as usize];
            world
                .process_at_rank(next_rank)
                .receive_into(&mut buffer[..]);
            received_points.append(&mut buffer);
        }

        // Filter out local points that's been sent to partner
        let mut points: Vec<Point> = points
            .iter()
            .filter(|&p| p.key >= min_seed)
            .cloned()
            .collect();

        received_points.append(&mut points);
        received_points.sort();

        received_points
    }

    /// Specialization for unbalanced trees.
    pub fn unbalanced_tree(
        world: &UserCommunicator,
        points: &[[PointType; 3]],
        domain: &Domain,
    ) -> (Vec<Point>, HashMap<MortonKey, MortonKey>) {
        let rank = world.rank();
        let size = world.size();

        // 1. Encode Points to Leaf Morton Keys, add a global index related to the processor
        let mut points: Vec<Point> = points
            .iter()
            .enumerate()
            .map(|(i, p)| Point {
                coordinate: *p,
                global_idx: i,
                key: MortonKey::from_point(p, domain),
            })
            .collect();

        // 2.i Perform parallel Morton sort over encoded points
        let comm = world.duplicate();
        hyksort(&mut points, K, comm);

        // 2.ii Find unique leaf keys on each processor and place in a Tree
        let keys: Vec<MortonKey> = points.iter().map(|p| p.key).collect();

        let mut tree = Tree { keys };

        // 3. Linearise received keys (remove overlaps if they exist).
        tree.linearize();

        // 4. Complete region spanned by node.
        tree.complete();

        // 5. Find seeds and compute the coarse blocktree
        let mut seeds = DistributedTree::find_seeds(&tree.keys);

        let blocktree = DistributedTree::complete_blocktree(&mut seeds, &rank, &size, world);

        // 5.ii any data below the min seed sent to partner process
        let points =
            DistributedTree::transfer_points_to_blocktree(world, &points, &seeds, &rank, &size);

        // 6. Refine blocks based on ncrit
        let map =
            DistributedTree::split_blocks(&points.iter().map(|p| p.key).collect(), blocktree.keys);

        (points, map)
    }

    /// Specialization for balanced trees.
    pub fn balanced_tree(
        world: &UserCommunicator,
        points: &[[PointType; 3]],
        domain: &Domain,
    ) -> (Vec<Point>, HashMap<MortonKey, MortonKey>) {
        // Create a distributed unbalanced tree;
        let rank = world.rank();
        let size = world.size();

        // 1. Encode Points to Leaf Morton Keys, add a global index related to the processor
        let mut points: Vec<Point> = points
            .iter()
            .enumerate()
            .map(|(i, p)| Point {
                coordinate: *p,
                global_idx: i,
                key: MortonKey::from_point(p, domain),
            })
            .collect();

        // 2.i Perform parallel Morton sort over encoded points
        let comm = world.duplicate();
        hyksort(&mut points, K, comm);

        // 2.ii Find unique leaf keys on each processor and place in a Tree
        let keys: Vec<MortonKey> = points.iter().map(|p| p.key).collect();

        let mut tree = Tree { keys };

        // 3. Linearise received keys (remove overlaps if they exist).
        tree.linearize();

        // 4. Complete region spanned by node.
        tree.complete();

        // 5.i Find seeds and compute the coarse blocktree
        let mut seeds = DistributedTree::find_seeds(&tree.keys);

        let blocktree = DistributedTree::complete_blocktree(&mut seeds, &rank, &size, world);

        // 5.ii Send data below the min seed sent to partner process
        let points =
            DistributedTree::transfer_points_to_blocktree(world, &points, &seeds, &rank, &size);

        // 6. Refine blocks based on ncrit
        let unbalanced_tree =
            DistributedTree::split_blocks(&points.iter().map(|p| p.key).collect(), blocktree.keys);

        // 7.i Create the minimal balanced tree for local octants, spanning the entire domain, and linearize
        let linearized = Tree {
            keys: unbalanced_tree
                .into_iter()
                .map(|(_, block)| block)
                .collect(),
        };
        linearized.balance();

        // 7.ii Assign the local leaves to the elements of this new balanced tree
        let points_to_balanced = DistributedTree::assign_nodes_to_leaves(
            &points.iter().map(|p| p.key).collect(),
            &linearized.keys,
        );

        let mut points: Vec<Point> = points
            .iter()
            .map(|p| Point {
                coordinate: p.coordinate,
                global_idx: p.global_idx,
                key: *points_to_balanced.get(&p.key).unwrap(),
            })
            .collect();

        // 8. Perform another distributed sort
        let comm = world.duplicate();
        hyksort(&mut points, K, comm);

        // 9. Remove local overlaps
        let keys: Vec<MortonKey> = points.iter().map(|p| p.key).collect();

        let mut tree = Tree { keys };
        tree.linearize();

        // 10. Map points to non-overlapping tree
        let map = DistributedTree::assign_nodes_to_leaves(
            &points.iter().map(|p| p.key).collect(),
            &tree.keys,
        );

        (points, map)
    }

    /// Read a DistributedTree from a from a HDF5 file on master process, and redistribute.
    pub fn read_hdf5(world: &UserCommunicator, filepath: String) -> DistributedTree {
        let comm = world.duplicate();
        let rank = comm.rank();
        let size = comm.size();

        let root_rank = 0;
        let root_process = comm.process_at_rank(root_rank);

        // Communicate point and key sizes
        let mut nlocal_keys = 0 as Count;
        let mut nlocal_points = 0 as Count;

        if rank == root_rank {
            let file = hdf5::File::open(&filepath).unwrap();
            let attr_names = file.attr_names().unwrap();
            let comm_size: Rank = file.attr("comm_size").unwrap().read_scalar().unwrap();

            // Communicator sizes have to match to replicate tree
            if comm_size != size {
                println!("Experiment size doesn't match world size!");
                comm.abort(1);
            } else {
                let global_key_counts: Vec<Count> = file
                    .dataset("key_counts")
                    .unwrap()
                    .read_raw::<Count>()
                    .unwrap();
                let global_point_counts: Vec<Count> = file
                    .dataset("point_counts")
                    .unwrap()
                    .read_raw::<Count>()
                    .unwrap();

                // println!("HERE {:?} ", global_point_counts);
                root_process.scatter_into_root(&global_key_counts, &mut nlocal_keys);
                root_process.scatter_into_root(&global_point_counts, &mut nlocal_points);
            }
        } else {
            root_process.scatter_into(&mut nlocal_keys);
            root_process.scatter_into(&mut nlocal_points);
        }

        // Communicate data
        let mut local_keys: Vec<MortonKey> = vec![MortonKey::default(); nlocal_keys as usize];
        let mut local_points: Vec<Point> = vec![Point::default(); nlocal_points as usize];

        // Placeholders for global values
        let mut balanced = bool::default();
        let mut global_domain = Domain::default();

        if rank == root_rank {
            let file = hdf5::File::open(&filepath).unwrap();
            let comm_size: Rank = file.attr("comm_size").unwrap().read_scalar().unwrap();

            // Communicator sizes have to match to replicate tree
            if comm_size != size {
                println!("Experiment size doesn't match world size!");
                comm.abort(1);
            } else {
                // Read global data into master process
                let global_keys: Vec<MortonKey> = file
                    .dataset("keys")
                    .unwrap()
                    .read_raw::<MortonKey>()
                    .unwrap();
                let global_key_counts: Vec<Count> = file
                    .dataset("key_counts")
                    .unwrap()
                    .read_raw::<Count>()
                    .unwrap();
                let global_key_displs: Vec<Count> = file
                    .dataset("key_displs")
                    .unwrap()
                    .read_raw::<Count>()
                    .unwrap();

                let global_points: Vec<Point> =
                    file.dataset("points").unwrap().read_raw::<Point>().unwrap();
                let global_point_counts: Vec<Count> = file
                    .dataset("point_counts")
                    .unwrap()
                    .read_raw::<Count>()
                    .unwrap();
                let global_point_displs: Vec<Count> = file
                    .dataset("point_displs")
                    .unwrap()
                    .read_raw::<Count>()
                    .unwrap();

                let domain = file.group("domain").unwrap();
                let origin: [PointType; 3] = domain
                    .dataset("origin")
                    .unwrap()
                    .read_raw::<PointType>()
                    .unwrap()[0..3]
                    .try_into()
                    .unwrap();
                let diameter: [PointType; 3] = domain
                    .dataset("diameter")
                    .unwrap()
                    .read_raw::<PointType>()
                    .unwrap()[0..3]
                    .try_into()
                    .unwrap();

                global_domain = Domain { origin, diameter };
                balanced = file.attr("balanced").unwrap().read_scalar().unwrap();

                // Distribute tree data to processes in communicator
                let key_partition =
                    Partition::new(&global_keys[..], global_key_counts, &global_key_displs[..]);

                let point_partition = Partition::new(
                    &global_points[..],
                    global_point_counts,
                    &global_point_displs[..],
                );

                root_process.scatter_varcount_into_root(&key_partition, &mut local_keys[..]);
                root_process.scatter_varcount_into_root(&point_partition, &mut local_points[..]);
            }
        } else {
            root_process.scatter_varcount_into(&mut local_keys[..]);
            root_process.scatter_varcount_into(&mut local_points[..]);
        }

        // Broadcast balance information
        root_process.broadcast_into(&mut balanced);

        // Broadcast domain
        root_process.broadcast_into(&mut global_domain);

        // Generate a local instance of distributed tree
        let points_to_keys = DistributedTree::assign_nodes_to_leaves(
            &local_points.iter().map(|p| p.key).collect(),
            &local_keys,
        );

        DistributedTree {
            keys: local_keys,
            points: local_points,
            points_to_keys: points_to_keys,
            balanced: balanced,
            domain: global_domain,
        }
    }

    /// Serialize a DistributedTree to HDF5.
    pub fn write_hdf5(
        world: &UserCommunicator,
        filename: String,
        tree: &DistributedTree,
    ) -> hdf5::Result<()> {
        // Communicate global data to root process
        let comm = world.duplicate();
        let rank = comm.rank();
        let size = comm.size();

        let root_rank = 0;
        let root_process = comm.process_at_rank(root_rank);

        // Gather the keys
        let local_keys = &tree.keys;
        let local_points = &tree.points;

        let nlocal_keys: Count = tree.keys.len() as Count;
        let nlocal_points: Count = tree.points.len() as Count;

        let mut global_key_counts: Vec<Count> = vec![0; size as usize];
        let mut global_point_counts: Vec<Count> = vec![0; size as usize];

        if rank == root_rank {
            root_process.gather_into_root(&nlocal_keys, &mut global_key_counts[..]);
            root_process.gather_into_root(&nlocal_points, &mut global_point_counts[..]);
        } else {
            root_process.gather_into(&nlocal_keys);
            root_process.gather_into(&nlocal_points);
        }

        // Write to file on root process
        if rank == root_rank {
            // Calculate point and key displacements
            let global_key_displs: Vec<Count> = global_key_counts
                .iter()
                .scan(0, |acc, &x| {
                    let tmp = *acc;
                    *acc += x;
                    Some(tmp)
                })
                .collect();

            let global_point_displs: Vec<Count> = global_point_counts
                .iter()
                .scan(0, |acc, &x| {
                    let tmp = *acc;
                    *acc += x;
                    Some(tmp)
                })
                .collect();

            // Buffer for global keys
            let global_key_count: usize = global_key_counts.iter().sum::<Count>() as usize;
            let mut global_keys: Vec<MortonKey> =
                vec![MortonKey::default(); global_key_count as usize];

            let mut key_partition = PartitionMut::new(
                &mut global_keys[..],
                global_key_counts.clone(),
                &global_key_displs[..],
            );
            root_process.gather_varcount_into_root(&local_keys[..], &mut key_partition);

            // Buffer for global points
            let global_point_count: usize = global_point_counts.iter().sum::<Count>() as usize;
            let mut global_points: Vec<Point> = vec![Point::default(); global_point_count as usize];

            let mut point_partition = PartitionMut::new(
                &mut global_points[..],
                global_point_counts.clone(),
                &global_point_displs[..],
            );
            root_process.gather_varcount_into_root(&local_points[..], &mut point_partition);

            // Write data
            {
                // Open file buffer
                let file = hdf5::File::create(format!("{}.hdf5", filename))?;

                // Write keys
                let keys = file
                    .new_dataset::<MortonKey>()
                    .shape([global_keys.len()])
                    .create("keys")?;
                keys.write(&global_keys)?;

                // Write points
                let points = file
                    .new_dataset::<Point>()
                    .shape([global_points.len()])
                    .create("points")?;
                points.write_raw(&global_points)?;

                // Write balance information as an attribute
                let attr_builder = file.new_attr::<bool>();
                let attr = attr_builder.create("balanced")?;
                attr.write_scalar(&tree.balanced)?;

                // Write world size as an attribute
                let attr_builder = file.new_attr::<i32>();
                let attr = attr_builder.create("comm_size")?;
                attr.write_scalar(&size)?;

                // Write key partition
                let key_counts = file
                    .new_dataset::<Count>()
                    .shape([global_key_counts.len()])
                    .create("key_counts")?;
                key_counts.write(&global_key_counts)?;

                let key_displs = file
                    .new_dataset::<Count>()
                    .shape([global_key_displs.len()])
                    .create("key_displs")?;
                key_displs.write(&global_key_displs)?;

                // Write point partition
                let point_counts = file
                    .new_dataset::<Count>()
                    .shape([global_point_counts.len()])
                    .create("point_counts")?;
                point_counts.write(&global_point_counts)?;

                let point_displs = file
                    .new_dataset::<Count>()
                    .shape([global_point_displs.len()])
                    .create("point_displs")?;
                point_displs.write(&global_point_displs)?;

                // Write domain
                let domain = file.create_group("domain")?;
                let origin = domain
                    .new_dataset::<PointType>()
                    .shape([3])
                    .create("origin")?;
                let diameter = domain
                    .new_dataset::<PointType>()
                    .shape([3])
                    .create("diameter")?;
                origin.write(&tree.domain.origin)?;
                diameter.write(&tree.domain.diameter)?;
            }
            Ok(())
        } else {
            root_process.gather_varcount_into(&local_keys[..]);
            root_process.gather_varcount_into(&local_points[..]);
            Ok(())
        }
    }

    /// Serialize a DistributedTree's keys to VTK for visualization.
    pub fn write_vtk(world: &UserCommunicator, filename: String, tree: &DistributedTree) {
        let comm = world.duplicate();
        let rank = comm.rank();
        let size = comm.size();

        // Communicate global leaves and global domain
        let root_rank = 0;
        let root_process = comm.process_at_rank(root_rank);

        // Gather the keys
        let local_keys = &tree.keys;

        let nlocal_keys: Count = tree.keys.len() as Count;

        let mut global_key_counts: Vec<Count> = vec![0; size as usize];

        if rank == root_rank {
            root_process.gather_into_root(&nlocal_keys, &mut global_key_counts[..]);
        } else {
            root_process.gather_into(&nlocal_keys);
        }

        // Write to file on root process
        if rank == root_rank {
            // Calculate point and key displacements
            let global_key_displs: Vec<Count> = global_key_counts
                .iter()
                .scan(0, |acc, &x| {
                    let tmp = *acc;
                    *acc += x;
                    Some(tmp)
                })
                .collect();

            // Buffer for global keys
            let global_key_count: usize = global_key_counts.iter().sum::<Count>() as usize;
            let mut global_keys: Vec<MortonKey> =
                vec![MortonKey::default(); global_key_count as usize];

            let mut key_partition = PartitionMut::new(
                &mut global_keys[..],
                global_key_counts.clone(),
                &global_key_displs[..],
            );
            root_process.gather_varcount_into_root(&local_keys[..], &mut key_partition);

            global_keys.write_vtk(filename, &tree.domain);
        } else {
            root_process.gather_varcount_into(&local_keys[..]);
        }
    }
}
