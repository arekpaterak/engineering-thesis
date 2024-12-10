include "globals.mzn";

int: nodes;
set of int: NUM = 1..nodes;
array[NUM,NUM] of float: dist;
array[NUM] of var NUM: route;
array[NUM] of 0..nodes: fixed_route;

var int: total_distance;

constraint alldifferent(route);
constraint route[1] = 1;

constraint total_distance = dist[route[nodes], 1] +
  sum([dist[route[i], route[i+1]] | i in NUM where i < nodes]);

constraint forall(n in NUM where fixed_route[n] != 0)(
  route[n] = fixed_route[n]
);

solve minimize total_distance;

output ["total distance: \(total_distance)\nroute = \(route)"];