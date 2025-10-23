from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import math
import time
import heapq

# Global caches
_distance_cache = {}
_path_cache = {}
_sku_cache = {}

def resolve_warehouse_node(env: LogisticsEnvironment, wh_or_node: Any) -> Optional[int]:
    """Convert a warehouse ID or node reference to a node ID."""
    if wh_or_node is None:
        return None
    if isinstance(wh_or_node, int):
        return wh_or_node
    if isinstance(wh_or_node, str) and wh_or_node in env.warehouses:
        wh = env.get_warehouse_by_id(wh_or_node)
        if wh and getattr(wh, "location", None):
            return int(wh.location.id)
    try:
        return int(wh_or_node)
    except Exception:
        return None


def get_order_node(env: LogisticsEnvironment, order_id: str) -> Optional[int]:
    """Get the node ID of an order's destination."""
    try:
        node = env.get_order_location(order_id)
        return int(node)
    except Exception:
        if order_id in env.orders:
            dest = env.orders[order_id].destination
            if dest and getattr(dest, "id", None) is not None:
                return int(dest.id)
    return None


def dijkstra_path(env: LogisticsEnvironment, start: int, target: int) -> Optional[Tuple[List[int], float]]:
    """Find shortest path using Dijkstra with caching."""
    cache_key = (start, target)
    if cache_key in _path_cache:
        return _path_cache[cache_key]
    
    if start == target:
        result = ([start], 0.0)
        _path_cache[cache_key] = result
        return result
    
    rn = env.get_road_network_data() or {}
    adjacency = rn.get("adjacency_list", {})
    
    pq = [(0.0, start, [start])]
    visited = set()
    
    while pq:
        dist, node, path = heapq.heappop(pq)
        
        if node in visited:
            continue
        visited.add(node)
        
        if node == target:
            result = (path, dist)
            _path_cache[cache_key] = result
            return result
        
        for neighbor_info in adjacency.get(node, []):
            if isinstance(neighbor_info, (list, tuple)) and len(neighbor_info) >= 2:
                neighbor, edge_dist = int(neighbor_info[0]), float(neighbor_info[1])
            else:
                neighbor = int(neighbor_info)
                edge_dist = 1.0
            
            if neighbor not in visited:
                heapq.heappush(pq, (dist + edge_dist, neighbor, path + [neighbor]))
    
    return None


def bfs_path(env: LogisticsEnvironment, start: int, target: int, max_expansions: int = 20000) -> Optional[List[int]]:
    """Find a connected path using BFS with caching."""
    result = dijkstra_path(env, start, target)
    return result[0] if result else None


def compute_order_weight_volume(env: LogisticsEnvironment, order_id: str) -> Tuple[float, float, Dict[str,int]]:
    """Calculate total weight and volume for an order with caching."""
    reqs = env.get_order_requirements(order_id) or {}
    total_w = 0.0
    total_v = 0.0
    
    for sku_id, qty in reqs.items():
        if sku_id not in _sku_cache:
            sku = env.get_sku_details(sku_id)
            if not sku:
                return float('inf'), float('inf'), reqs
            _sku_cache[sku_id] = (sku.get('weight', 0.0), sku.get('volume', 0.0))
        
        unit_w, unit_v = _sku_cache[sku_id]
        total_w += unit_w * qty
        total_v += unit_v * qty
    
    return total_w, total_v, reqs


def get_candidate_warehouses_for_sku(env: LogisticsEnvironment, sku_id: str, min_qty: int = 1) -> List[str]:
    """Return a list of warehouses that have at least min_qty of a SKU."""
    try:
        whs = env.get_warehouses_with_sku(sku_id, min_quantity=min_qty) or []
        return list(whs)
    except TypeError:
        try:
            whs = env.get_warehouses_with_sku(sku_id) or []
            return list(whs)
        except Exception:
            return []


def route_distance_using_env(env: LogisticsEnvironment, steps: List[Dict]) -> float:
    """Compute total distance between nodes in a route with caching."""
    if not steps:
        return 0.0
    
    nodes = []
    for s in steps:
        nid = int(s['node_id'])
        if not nodes or nodes[-1] != nid:
            nodes.append(nid)
    
    total = 0.0
    for i in range(len(nodes)-1):
        a, b = nodes[i], nodes[i+1]
        cache_key = (a, b)
        
        if cache_key in _distance_cache:
            total += _distance_cache[cache_key]
            continue
        
        result = dijkstra_path(env, a, b)
        if result:
            _, d = result
            _distance_cache[cache_key] = float(d)
            total += float(d)
        else:
            return float('inf')
    
    return total


DEFAULT_SPECS = {
    "Light_Item": (5.0, 0.02),
    "Medium_Item": (15.0, 0.06),
    "Heavy_Item": (30.0, 0.12),
}

def get_wv(env, sku_id):
    """Get weight and volume with caching."""
    if sku_id not in _sku_cache:
        sku = env.get_sku_details(sku_id)
        if sku and "weight" in sku and "volume" in sku:
            _sku_cache[sku_id] = (float(sku["weight"]), float(sku["volume"]))
        else:
            _sku_cache[sku_id] = DEFAULT_SPECS.get(sku_id, (5.0, 0.02))
    return _sku_cache[sku_id]


def estimate_distance(env: LogisticsEnvironment, node1: int, node2: int) -> float:
    """Estimate distance between nodes using haversine or cached distance."""
    cache_key = (node1, node2)
    if cache_key in _distance_cache:
        return _distance_cache[cache_key]
    
    try:
        n1 = env.nodes.get(node1)
        n2 = env.nodes.get(node2)
        if n1 and n2 and hasattr(n1, 'lat') and hasattr(n2, 'lat'):
            # Haversine approximation
            lat_diff = (n1.lat - n2.lat) * 111.0  # km per degree
            lon_diff = (n1.lon - n2.lon) * 111.0 * math.cos(math.radians((n1.lat + n2.lat) / 2))
            dist = math.sqrt(lat_diff**2 + lon_diff**2)
            return dist
    except Exception:
        pass
    
    return 0.0


def find_best_warehouses(env: LogisticsEnvironment, order_id: str, reqs: Dict[str, int],
                         local_inventory: Dict[str, Dict[str, int]],
                         order_node: int) -> Dict[str, Dict[str, int]]:
    """Find optimal warehouse allocation for order requirements."""
    pickup_plan = {}
    
    for sku_id, qty_needed in reqs.items():
        candidates = get_candidate_warehouses_for_sku(env, sku_id, min_qty=1)
        
        # Score warehouses by distance and availability
        scored_whs = []
        for wh_id in candidates:
            available = local_inventory.get(wh_id, {}).get(sku_id, 0)
            if available <= 0:
                continue
            
            wh = env.get_warehouse_by_id(wh_id)
            if not wh or not getattr(wh, 'location', None):
                continue
            
            wh_node = int(wh.location.id)
            dist = estimate_distance(env, wh_node, order_node)
            
            # Prefer warehouses with more stock and closer distance
            score = -dist + (available * 0.1)
            scored_whs.append((score, wh_id, available))
        
        scored_whs.sort(reverse=True)
        
        # Allocate from best warehouses
        needed = qty_needed
        for _, wh_id, available in scored_whs:
            if needed <= 0:
                break
            take = min(available, needed)
            pickup_plan.setdefault(wh_id, {})
            pickup_plan[wh_id][sku_id] = pickup_plan[wh_id].get(sku_id, 0) + take
            needed -= take
        
        if needed > 0:
            return {}  # Cannot fulfill this order
    
    return pickup_plan


def optimize_delivery_sequence(env: LogisticsEnvironment, home_node: int, 
                               order_nodes: List[int]) -> List[int]:
    """Optimize delivery sequence using nearest neighbor heuristic."""
    if not order_nodes:
        return []
    
    unvisited = set(order_nodes)
    sequence = []
    current = home_node
    
    while unvisited:
        nearest = min(unvisited, key=lambda n: estimate_distance(env, current, n))
        sequence.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    return sequence


def plan_optimized_trip(env: LogisticsEnvironment,
                       vehicle_obj,
                       orders_with_data: List[Tuple[str, float, float, Dict[str, int], int]],
                       local_inventory: Dict[str, Dict[str, int]],
                       warehouse_nodes: Dict[str, int],
                       max_distance: float) -> Tuple[List[Dict], List[str], float]:
    """Plan one optimized trip with better routing and load optimization."""
    home_node = warehouse_nodes.get(vehicle_obj.id)
    if home_node is None:
        return [], [], 0.0

    max_w = float(getattr(vehicle_obj, 'capacity_weight', 0.0))
    max_v = float(getattr(vehicle_obj, 'capacity_volume', 0.0))
    
    chosen_orders = []
    pickup_plan = {}
    total_weight = 0.0
    total_volume = 0.0
    
    # Select orders using knapsack-like approach
    for order_id, ow, ov, reqs, order_node in orders_with_data:
        if total_weight + ow > max_w or total_volume + ov > max_v:
            continue
        
        # Find optimal warehouse allocation
        order_pickups = find_best_warehouses(env, order_id, reqs, local_inventory, order_node)
        
        if not order_pickups:
            continue
        
        # Check if adding this order exceeds distance constraint
        temp_orders = chosen_orders + [order_id]
        temp_nodes = [order_node for oid, _, _, _, on in orders_with_data if oid in temp_orders for on in [on]]
        
        # Estimate trip distance
        est_dist = 0.0
        for wh_id in order_pickups.keys():
            wh = env.get_warehouse_by_id(wh_id)
            if wh and getattr(wh, 'location', None):
                wh_node = int(wh.location.id)
                est_dist += estimate_distance(env, home_node, wh_node)
        
        for node in temp_nodes:
            est_dist += estimate_distance(env, home_node, node)
        
        if max_distance > 0 and est_dist > max_distance * 0.8:  # Safety margin
            continue
        
        # Accept this order
        chosen_orders.append(order_id)
        total_weight += ow
        total_volume += ov
        
        for wh_id, skumap in order_pickups.items():
            pickup_plan.setdefault(wh_id, {})
            for sku_id, q in skumap.items():
                pickup_plan[wh_id][sku_id] = pickup_plan[wh_id].get(sku_id, 0) + q
                local_inventory[wh_id][sku_id] -= q
    
    if not chosen_orders:
        return [], [], 0.0
    
    # Build optimized route
    steps = []
    current_node = home_node
    
    # Optimize warehouse visit sequence
    wh_nodes = []
    wh_mapping = {}
    for wh_id in pickup_plan.keys():
        wh = env.get_warehouse_by_id(wh_id)
        if wh and getattr(wh, 'location', None):
            wh_node = int(wh.location.id)
            wh_nodes.append(wh_node)
            wh_mapping[wh_node] = wh_id
    
    wh_sequence = optimize_delivery_sequence(env, home_node, wh_nodes)
    
    # Pickup phase
    for wh_node in wh_sequence:
        wh_id = wh_mapping[wh_node]
        path = bfs_path(env, current_node, wh_node)
        if not path:
            continue
        
        for node in path[:-1]:
            steps.append({'node_id': int(node), 'pickups': [], 'deliveries': [], 'unloads': []})
        
        pickups_list = []
        for sku_id, q in pickup_plan[wh_id].items():
            pickups_list.append({'warehouse_id': wh_id, 'sku_id': sku_id, 'quantity': int(q)})
        
        steps.append({'node_id': int(wh_node), 'pickups': pickups_list, 'deliveries': [], 'unloads': []})
        current_node = wh_node
    
    # Optimize delivery sequence
    order_node_map = {oid: on for oid, _, _, _, on in orders_with_data if oid in chosen_orders}
    delivery_nodes = list(order_node_map.values())
    optimized_sequence = optimize_delivery_sequence(env, current_node, delivery_nodes)
    
    # Delivery phase
    for delivery_node in optimized_sequence:
        order_ids = [oid for oid, on in order_node_map.items() if on == delivery_node]
        
        path = bfs_path(env, current_node, delivery_node)
        if not path:
            continue
        
        for node in path[:-1]:
            steps.append({'node_id': int(node), 'pickups': [], 'deliveries': [], 'unloads': []})
        
        deliveries = []
        for oid in order_ids:
            reqs = env.get_order_requirements(oid)
            for sku, q in reqs.items():
                deliveries.append({'order_id': oid, 'sku_id': sku, 'quantity': q})
        
        steps.append({'node_id': int(delivery_node), 'pickups': [], 'deliveries': deliveries, 'unloads': []})
        current_node = delivery_node
    
    # Return home
    path_home = bfs_path(env, current_node, home_node)
    if path_home:
        for node in path_home[1:]:
            steps.append({'node_id': int(node), 'pickups': [], 'deliveries': [], 'unloads': []})
    
    actual_distance = route_distance_using_env(env, steps)
    
    return steps, chosen_orders, actual_distance


def solver(env: LogisticsEnvironment) -> Dict:
    """Advanced MWVRP solver with multi-phase optimization."""
    solution = {'routes': []}
    start_time = time.time()
    
    # Clear caches
    _distance_cache.clear()
    _path_cache.clear()
    _sku_cache.clear()
    
    all_orders = env.get_all_order_ids() or []
    available_vehicle_ids = env.get_available_vehicles() or []
    
    # Get vehicles
    vehicles = []
    for vid in available_vehicle_ids:
        try:
            vehicles.append(env.get_vehicle_by_id(vid))
        except Exception:
            for v in env.get_all_vehicles():
                if getattr(v, 'id', None) == vid:
                    vehicles.append(v)
                    break
    
    # Sort vehicles by capacity (largest first)
    vehicles.sort(key=lambda v: (
        getattr(v, 'capacity_weight', 0) * getattr(v, 'capacity_volume', 0)
    ), reverse=True)
    
    # Pre-compute data
    warehouse_nodes = {}
    for v in vehicles:
        home_wh_id = getattr(v, "home_warehouse_id", None)
        home_node = resolve_warehouse_node(env, home_wh_id) or resolve_warehouse_node(env, env.get_vehicle_home_warehouse(v.id))
        warehouse_nodes[v.id] = home_node
    
    order_nodes = {}
    for order_id in all_orders:
        order_nodes[order_id] = get_order_node(env, order_id)
    
    # Initialize inventory
    local_inventory = {}
    for wh_id, wh in env.warehouses.items():
        inv = env.get_warehouse_inventory(wh_id) or {}
        local_inventory[wh_id] = {sku: int(q) for sku, q in inv.items()}
    
    # Pre-compute order data
    order_data = []
    for order_id in all_orders:
        ow, ov, reqs = compute_order_weight_volume(env, order_id)
        if ow == float('inf'):
            continue
        onode = order_nodes.get(order_id)
        if onode is None:
            continue
        order_data.append((order_id, ow, ov, reqs, onode))
    
    # Sort orders by weight/volume ratio (pack efficiently)
    order_data.sort(key=lambda x: -(x[1] / max(x[2], 0.01)))
    
    unassigned = set(oid for oid, _, _, _, _ in order_data)
    vehicle_routes = {v.id: [] for v in vehicles}
    vehicle_distances = {v.id: 0.0 for v in vehicles}
    
    # Main allocation loop
    MAX_ITERATIONS = 100
    iteration = 0
    
    while unassigned and iteration < MAX_ITERATIONS:
        iteration += 1
        progress = False
        
        for v in vehicles:
            if not unassigned:
                break
            
            max_dist = getattr(v, 'max_distance', 0.0)
            remaining_dist = max_dist - vehicle_distances[v.id] if max_dist > 0 else float('inf')
            
            if remaining_dist <= 0:
                continue
            
            # Select orders for this vehicle
            available_orders = [od for od in order_data if od[0] in unassigned]
            
            steps, assigned, trip_dist = plan_optimized_trip(
                env, v, available_orders, local_inventory, 
                warehouse_nodes, remaining_dist
            )
            
            if steps and assigned:
                vehicle_routes[v.id].extend(steps)
                vehicle_distances[v.id] += trip_dist
                
                for oid in assigned:
                    unassigned.discard(oid)
                
                progress = True
        
        if not progress:
            break
    
    # Emergency allocation for remaining orders
    if unassigned:
        print(f"⚠️  Attempting emergency allocation for {len(unassigned)} orders")
        for oid in list(unassigned):
            od = next((x for x in order_data if x[0] == oid), None)
            if not od:
                continue
            
            for v in vehicles:
                max_dist = getattr(v, 'max_distance', 0.0)
                remaining_dist = max_dist - vehicle_distances[v.id] if max_dist > 0 else float('inf')
                
                steps, assigned, trip_dist = plan_optimized_trip(
                    env, v, [od], local_inventory,
                    warehouse_nodes, remaining_dist
                )
                
                if steps and assigned:
                    vehicle_routes[v.id].extend(steps)
                    vehicle_distances[v.id] += trip_dist
                    unassigned.discard(oid)
                    break
    
    # Finalize routes
    for v in vehicles:
        r = vehicle_routes.get(v.id, [])
        if not r:
            continue
        
        home_node = warehouse_nodes[v.id]
        
        # Ensure route starts and ends at home
        if int(r[0]['node_id']) != home_node:
            r.insert(0, {'node_id': int(home_node), 'pickups': [], 'deliveries': [], 'unloads': []})
        
        if int(r[-1]['node_id']) != home_node:
            r.append({'node_id': int(home_node), 'pickups': [], 'deliveries': [], 'unloads': []})
        
        solution['routes'].append({'vehicle_id': v.id, 'steps': r})
    
    elapsed = time.time() - start_time
    fulfilled = len(all_orders) - len(unassigned)
    
    print("=" * 70)
    print(f"🚚 MWVRP SOLVER RESULTS")
    print("=" * 70)
    print(f"⏱️  Execution Time: {elapsed:.2f}s")
    print(f"📦 Orders Fulfilled: {fulfilled}/{len(all_orders)} ({100*fulfilled/len(all_orders):.1f}%)")
    print(f"🚛 Vehicles Used: {len(solution['routes'])}/{len(vehicles)}")
    
    if unassigned:
        print(f"⚠️  Unfulfilled Orders ({len(unassigned)}): {list(unassigned)[:10]}")
    
    for v in vehicles:
        if v.id in vehicle_distances and vehicle_distances[v.id] > 0:
            max_d = getattr(v, 'max_distance', 0.0)
            dist_pct = (vehicle_distances[v.id] / max_d * 100) if max_d > 0 else 0
            print(f"   {v.id}: {vehicle_distances[v.id]:.1f}km / {max_d:.1f}km ({dist_pct:.1f}%)")
    
    print("=" * 70)
    
    return solution

# if __name__ == "__main__":
#     env = LogisticsEnvironment()
#     sol = solver(env)
#     valid, msg = env.validate_solution_complete(sol)
#     print("Validation:", valid, msg)
#     print("Routes returned:", len(sol.get("routes", [])))
