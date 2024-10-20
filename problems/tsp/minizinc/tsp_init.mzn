include "globals.mzn";

int: Nodes;
set of int: NUM = 1..Nodes;
array[NUM,NUM] of int: Dist;
array[NUM] of var NUM: route;

var int: total_distance;

constraint alldifferent(route);
constraint route[1] = 1;

constraint total_distance = Dist[route[Nodes], 1] + 
    sum([Dist[route[i], route[i+1]] | i in NUM where i < Nodes]);

solve satisfy;

output ["total_distance: \(total_distance)\nroute = \(route)"];