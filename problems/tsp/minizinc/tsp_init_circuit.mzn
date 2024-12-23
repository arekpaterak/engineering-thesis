include "globals.mzn";

int: nodes;  % Number of nodes
set of int: NUM = 1..nodes;
array[NUM, NUM] of int: dist;  % Distance matrix

var int: total_distance;
array[NUM] of var NUM: next;  % Represents the next node in the route

% Ensure the sequence forms a valid Hamiltonian circuit
constraint circuit(next);

% Compute the total distance for the route
constraint total_distance =
    sum([dist[i, next[i]] | i in NUM]);

solve satisfy;

output ["total_distance: \(total_distance)\nnext = \(next)"];
