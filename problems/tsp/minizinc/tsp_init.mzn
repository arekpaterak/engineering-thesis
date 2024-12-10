include "globals.mzn";

int: nodes;
set of int: NUM = 1..nodes;
array[NUM,NUM] of int: dist;
array[NUM] of var NUM: route;

var int: total_distance;

constraint alldifferent(route);
constraint route[1] = 1;

constraint total_distance = dist[route[nodes], 1] +
    sum([dist[route[i], route[i+1]] | i in NUM where i < nodes]);

solve satisfy;

output ["total_distance: \(total_distance)\nroute = \(route)"];