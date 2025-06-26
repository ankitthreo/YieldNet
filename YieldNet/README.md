# chemprop

The features added and changes made in the original are summarized and briefly explained
here. The arguments that have been added cover all variants tried.

## Execution
1. dataset-path: the model can run on one triplet of data at a time - `train.csv`, `val.csv`
and `test.csv`, and the paths for these files needs to be mentioned in the run_one.sh
script. For example, if the dataset folder is such that it has the location for one set of csv as follows: `./dataset/gp0/train/cv1.csv` (and corresponding folders for val and
test), then the paths for train, val and test in the run_one script will have to be set as per this (if the numbers 0 and 1 in the path are variable, they can be taken through command line arguments).
2. execution: for running the model on the entire dataset, the rand_run.py script can
be used, which internally uses the `run_one.sh` script on each of the triplets of files.

## Arguments
- `use_lrl_network`: boolean for the LRL network that is used as part of the MPN Encoder.
The network maps the messages obtained from reactant and product to create pseudo
features for bonds, which is then used to calculate adjacency matrices specifically
for features of reactant and product. By default LRL is not used.
- `format`: Literal indicating which type of atom alignment training is to be done (by
defualt it is `"perm"` i.e. permutation). Other option is `"att"` for attention.
- `perm_type`: Literal indicating which type of permutation training is to be done (by
defualt it is soft permutation).
- `similarity`: Literal indicating which type of cost matrix should be performed (by
defualt it is `"max"`). Alternative is `"dot"`
- `gumbel_noise_factor`: float value for the factor with which the sample noise from
the Gumbel distribution has to be scaled before adding at start of Sinkhorn iterations.
The default value is `1.0`.
- `gumbel_temperature`: float value for the temperature used in Sinkhorn iterations. The
default value is `0.1`.
- `sinkhorn_iters`: int value for the number of iterations in Sinkhorn. The default value
is `10`.
- `use_indicator`: boolean for using the indicator matrix along with the permutation. This
is a matrix which indicates whether atom at index "i" in the reactant is the same (by
atomic number) as the atom at index "j" in the product. Hence when this matrix is used
as a mask for the permutation of atoms in reactant-product adjacency matrix, it will
cause the places where the atoms are unequal to become 0. By default indicator matrix
is not used.
- `perm_regularizer`: Literal indicating the regularizer to be used along with permutation.
There are several regularizers already implemented. The functional forms of each are in
the file REGULARIZERS.txt as well. By default, the `'squared_fro_norm_inv'` is used.
- `perm_reg_lambda`: float value for the lambda to be used along with the regularizer loss.
This hyper-parameter can be tuned as needed. The permutation loss is added to the model
MSE loss after scaling by lambda. The default value is `0.01`.
- `loss_agg_func`: Literal indicating the aggregator to be used for the MSE loss, before
adding the permutation loss. By default the `'sum'` is used.
- `use_node_tf`: boolean indicating whether to use the transformer on node embeddings or not.
Default is `False`.
- `node_aggregation`: `Literal['set2set', 'deepset', 'mean', 'sum', 'norm']`. Default is `'mean'`
- `step_aggregation`: `Literal['set2set', 'deepset', 'capsule', 'sum']`. Default is `'set2set'`.

## Featurization
- random shuffling: a random shuffling of the reactant to product mapping is done for the
product indices. For example, {1:1,2:3,3:2} could change to {1:3,2:2,3:1}. The seed used
in this process is `R2P_SEED` (value 20).
- permutation matrix: If `perm_type` is hard, then use permutation matrix to calculate
hard permutation matrix.
- adjacency matrix: Adjacency matrices of the reactant and product have been calculated
inside `MolGraph` class to proceed tensorization and network based on adjacency.
- untangled features: reactant and product features are untangled and calculated
separately
without using `R2P` mapping.
- padding: to promote tensorization, we used padding with zero tensors within batches
with same number of nodes in BatchMolGraph
- indicator:
- tensorization: features, adjacencies, permutation matrices all are tensorized, where
the 0th dimension is batch size (`B`). if N is the number of nodes, Dn is the node feature
depth, De is edge feature depth, then node features have shape of `(B, N, Dn)`, edge
features `(B, N, N, De)`, permutation, indicator, adjacencies have shape of `(B, N, N)`. 

## MPN
- `permutation` = `hard_perm` if `perm_type` is `'hard'` else it is `soft_perm`
- `soft_perm`: permutation is produced by Gumble-Sinkhorn trick from reactant and product 
node hidden state (outputs of GNN for reactant and product).
- adjacency: as our approach is adjacency based, we need adjacency metrics for the
combined graph. Thus, adjacency of combined graph, A_u = max(A_r*M_r,P(A_p * M_p)P^{T}).
where A_r and A_p are the adjacency of reactant (`adj_r`, in code) and product(`adj_p`)
respectively. M_r (`message_r`) and M_p (`message_p`) are the messages collected from same 
GNN for both reactant and product. 
- tensorization: we did use only adjacency metrices for graph operations. Thus for
creation of edge feature matrices and messages we have multiply our edge features and
messages with adjacencies appropriately, so that we can features or message only for
such pair where we have an edge.  
- `LRL`: as we multiplied (Hadamard) our adjacency with message to make it more soft and
later we used the adjacency for tensorized graph operation, adjacency should match dims
with message or features. For message it is obvious. But soft adjaceny, a Hadamard
product of binary adjacency and message will not same dims with features. Thus there are
two alternatives-- one (when `use_lrl_network==True`), using Linear_ReLu_Linear(LRL)
network to reduce the message dims to feature dims then multiplied with binary adjcency
and two(when `use_lrl_network==False`), instead of using message use feature for
multiplication with binary adjacency.
- `indicator`: this matrix is an indicator function of the shape of the adjacency matrix
defined such that for the ith atom in the reactant (row) and jth atom in the product
(column) the matrix is 1 if the atoms are the same (as per atomic number) else 0 (the
zero-atom is also treated to be equal to another zero-atom, this arises due to padding)
- `perm_loss`: this function is to penalise the permutation matrix calculated from the
sinkhorn iterations in a way such that two atoms of the same type can be shuffled in
the product while two atoms of different types (atomic numbers) should not be shuffled
