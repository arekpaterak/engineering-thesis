include "globals.mzn";

int: nodes;  % Number of nodes
set of int: NUM = 1..nodes;
array[NUM, NUM] of float: dist;  % Distance matrix
array[NUM] of 0..nodes: fixed_next;  % The fixed next nodes (0 means no fixed node)

var int: total_distance;
array[NUM] of var NUM: next;  % Represents the next node in the route

% Ensure the sequence forms a valid Hamiltonian circuit
constraint circuit(next);

constraint forall(i in NUM where fixed_next[i] != 0)(
    next[i] = fixed_next[i]
);

constraint total_distance =
    sum([dist[i, next[i]] | i in NUM]);

solve minimize total_distance;

output ["total distance: \(total_distance)\nnext = \(next)"];
