include "globals.mzn";

int: Nodes;
set of int: NUM = 1..Nodes;
array[NUM,NUM] of float: Dist;
array[NUM] of var NUM: route;
array[NUM] of 0..Nodes: fixed_route;

var int: total_distance;

constraint alldifferent(route);
constraint route[1] = 1;

constraint total_distance = Dist[route[Nodes], 1] + 
  sum([Dist[route[i], route[i+1]] | i in NUM where i < Nodes]);

constraint forall(n in NUM where fixed_route[n] != 0)(
  route[n] = fixed_route[n]
);

solve minimize total_distance;

output ["total distance: \(total_distance)\nroute = \(route)"];