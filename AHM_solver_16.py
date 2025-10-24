from robin_logistics import LogisticsEnvironment
from typing import Dict, List, Optional, Tuple, Any
import math
import time
import heapq

# --- Caches ---
_distance_cache: Dict[Tuple[int, int], float] = {}
_path_cache: Dict[Tuple[int, int], Tuple[List[int], float]] = {}
_sku_cache: Dict[str, Tuple[float, float]] = {}

# --- Helpers ---
def resolve_warehouse_node(env: LogisticsEnvironment, wh_or_node: Any) -> Optional[int]:
    if wh_or_node is None:
        return None
    if isinstance(wh_or_node, int):
        return wh_or_node
    if isinstance(wh_or_node, str) and wh_or_node in env.warehouses:
        wh = env.get_warehouse_by_id(wh_or_node)
        if wh and getattr(wh, "location", None):
            try:
                return int(wh.location.id)
            except Exception:
                try:
                    return int(str(wh.location.id))
                except Exception:
                    return None
    try:
        return int(wh_or_node)
    except Exception:
        return None

def get_order_node(env: LogisticsEnvironment, order_id: str) -> Optional[int]:
    try:
        node = env.get_order_location(order_id)
        return int(node)
    except Exception:
        if order_id in env.orders:
            dest = env.orders[order_id].destination
            if dest and getattr(dest, "id", None) is not None:
                return int(dest.id)
    return None

def _normalize_adjacency(raw_adj: Any) -> Dict[int, List[Tuple[int, float]]]:
    adjacency: Dict[int, List[Tuple[int, float]]] = {}
    if not raw_adj:
        return adjacency
    for k, neighs in raw_adj.items():
        try:
            ik = int(k)
        except Exception:
            continue
        if not neighs:
            adjacency.setdefault(ik, [])
            continue
        adjacency.setdefault(ik, [])
        for e in neighs:
            try:
                if isinstance(e, dict):
                    nid = e.get("to") or e.get("node") or e.get("id")
                    dist = e.get("dist") or e.get("weight") or e.get("distance") or 1.0
                    adjacency[ik].append((int(nid), float(dist)))
                elif isinstance(e, (list, tuple)) and len(e) >= 2:
                    adjacency[ik].append((int(e[0]), float(e[1])))
                else:
                    adjacency[ik].append((int(e), 1.0))
            except Exception:
                continue
    return adjacency

def estimate_distance(env: LogisticsEnvironment, node1: int, node2: int) -> float:
    key = (node1, node2)
    if key in _distance_cache:
        return _distance_cache[key]
    try:
        n1 = env.nodes.get(node1)
        n2 = env.nodes.get(node2)
        if n1 and n2 and getattr(n1, 'lat', None) is not None and getattr(n2, 'lat', None) is not None:
            lat1, lon1 = float(n1.lat), float(n1.lon)
            lat2, lon2 = float(n2.lat), float(n2.lon)
            r = 6371.0
            phi1, phi2 = math.radians(lat1), math.radians(lat2)
            dphi = math.radians(lat2 - lat1)
            dlambda = math.radians(lon2 - lon1)
            a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
            dist = 2 * r * math.asin(min(1, math.sqrt(a)))
            _distance_cache[key] = dist
            return dist
    except Exception:
        pass
    _distance_cache[key] = 0.001
    return 0.001

# --- A* pathfinder ---
def a_star_path(env: LogisticsEnvironment, start: int, target: int) -> Optional[Tuple[List[int], float]]:
    key = (start, target)
    if key in _path_cache:
        return _path_cache[key]
    if start == target:
        _path_cache[key] = ([start], 0.0)
        return _path_cache[key]

    rn = env.get_road_network_data() or {}
    raw_adj = rn.get("adjacency_list", {}) or {}
    adjacency = _normalize_adjacency(raw_adj)

    h_start = estimate_distance(env, start, target)
    open_heap = [(h_start, 0.0, start, [start])]
    g_scores = {start: 0.0}
    closed = set()

    while open_heap:
        f, g, node, path = heapq.heappop(open_heap)
        if node in closed:
            continue
        if node == target:
            _path_cache[key] = (path, g)
            return _path_cache[key]
        closed.add(node)
        for neighbor, w in adjacency.get(node, []):
            tentative_g = g + float(w)
            if neighbor in closed:
                continue
            if tentative_g < g_scores.get(neighbor, float('inf')):
                g_scores[neighbor] = tentative_g
                h = estimate_distance(env, neighbor, target)
                heapq.heappush(open_heap, (tentative_g + h, tentative_g, neighbor, path + [neighbor]))

    return None

# --- Distance helpers ---
def segment_distance(env: LogisticsEnvironment, a: int, b: int) -> Optional[float]:
    try:
        if hasattr(env, 'get_distance'):
            d = env.get_distance(a, b)
            if d is not None:
                d = float(d)
                _distance_cache[(a, b)] = d
                return d
    except Exception:
        pass

    if (a, b) in _distance_cache:
        return _distance_cache[(a, b)]
    if (b, a) in _distance_cache:
        return _distance_cache[(b, a)]

    path_res = a_star_path(env, a, b)
    if path_res:
        _, d = path_res
        _distance_cache[(a, b)] = float(d)
        return float(d)

    est = estimate_distance(env, a, b)
    if est > 0:
        _distance_cache[(a, b)] = est
        return est

    return None

def route_distance_using_env(env: LogisticsEnvironment, steps: List[Dict]) -> float:
    if not steps:
        return 0.0

    nodes = []
    for s in steps:
        nid = int(s['node_id'])
        if not nodes or nodes[-1] != nid:
            nodes.append(nid)

    try:
        if hasattr(env, 'get_route_distance'):
            val = env.get_route_distance(nodes)
            if isinstance(val, (int, float)):
                tot = float(val)
                if tot <= 0 and len(nodes) > 1:
                    tot_fallback = 0.0
                    for i in range(len(nodes) - 1):
                        sd = segment_distance(env, nodes[i], nodes[i+1])
                        if sd is None:
                            return float('inf')
                        tot_fallback += sd
                    return tot_fallback
                return tot
    except Exception:
        pass

    total = 0.0
    for i in range(len(nodes) - 1):
        sd = segment_distance(env, nodes[i], nodes[i+1])
        if sd is None:
            return float('inf')
        total += sd

    if total <= 0:
        return float('inf')
    return float(total)

# --- SKU / order helpers ---
DEFAULT_SPECS = {
    "Light_Item": (5.0, 0.02),
    "Medium_Item": (15.0, 0.06),
    "Heavy_Item": (30.0, 0.12),
}

def get_wv(env: LogisticsEnvironment, sku_id: str) -> Tuple[float, float]:
    if sku_id not in _sku_cache:
        sku = env.get_sku_details(sku_id)
        if sku and "weight" in sku and "volume" in sku:
            _sku_cache[sku_id] = (float(sku["weight"]), float(sku["volume"]))
        else:
            _sku_cache[sku_id] = DEFAULT_SPECS.get(sku_id, (5.0, 0.02))
    return _sku_cache[sku_id]

def compute_order_weight_volume(env: LogisticsEnvironment, order_id: str) -> Tuple[float, float, Dict[str,int]]:
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
        total_w += float(unit_w) * int(qty)
        total_v += float(unit_v) * int(qty)
    return total_w, total_v, reqs

def get_candidate_warehouses_for_sku(env: LogisticsEnvironment, sku_id: str, min_qty: int = 1) -> List[str]:
    try:
        whs = env.get_warehouses_with_sku(sku_id, min_quantity=min_qty) or []
        return list(whs)
    except TypeError:
        try:
            whs = env.get_warehouses_with_sku(sku_id) or []
            return list(whs)
        except Exception:
            return []

def find_best_warehouses(env: LogisticsEnvironment, order_id: str, reqs: Dict[str, int],
                         local_inventory: Dict[str, Dict[str, int]], order_node: int) -> Dict[str, Dict[str, int]]:
    pickup_plan: Dict[str, Dict[str, int]] = {}
    for sku_id, qty_needed in reqs.items():
        candidates = get_candidate_warehouses_for_sku(env, sku_id, min_qty=1)
        scored = []
        for wh_id in candidates:
            available = local_inventory.get(wh_id, {}).get(sku_id, 0)
            if available <= 0:
                continue
            wh = env.get_warehouse_by_id(wh_id)
            if not wh or not getattr(wh, 'location', None):
                continue
            try:
                wh_node = int(wh.location.id)
            except Exception:
                continue
            dist = estimate_distance(env, wh_node, order_node)
            score = -dist + (available * 0.05)
            scored.append((score, wh_id, available))
        scored.sort(reverse=True)
        needed = int(qty_needed)
        for _, wh_id, avail in scored:
            if needed <= 0:
                break
            take = min(avail, needed)
            pickup_plan.setdefault(wh_id, {})
            pickup_plan[wh_id][sku_id] = pickup_plan[wh_id].get(sku_id, 0) + int(take)
            needed -= take
        if needed > 0:
            return {}
    return pickup_plan

def optimize_delivery_sequence(env: LogisticsEnvironment, home_node: int, order_nodes: List[int]) -> List[int]:
    if not order_nodes:
        return []
    unvisited = set(order_nodes)
    seq: List[int] = []
    cur = home_node
    while unvisited:
        nearest = min(unvisited, key=lambda n: estimate_distance(env, cur, n))
        seq.append(nearest)
        unvisited.remove(nearest)
        cur = nearest
    return seq

# helper to merge operation lists (used by dedup merging)
def _merge_ops(target_step: Dict, source_step: Dict) -> None:
    """Merge pickups/deliveries/unloads from source_step into target_step (in-place)."""
    # pickups: list of {'warehouse_id', 'sku_id', 'quantity'}
    for p in source_step.get('pickups', []) or []:
        # try to combine quantities if same warehouse+sku exists
        matched = False
        for tp in target_step.setdefault('pickups', []):
            if tp.get('warehouse_id') == p.get('warehouse_id') and tp.get('sku_id') == p.get('sku_id'):
                tp['quantity'] = tp.get('quantity', 0) + p.get('quantity', 0)
                matched = True
                break
        if not matched:
            target_step['pickups'].append(p.copy())
    # deliveries: list of {'order_id','sku_id','quantity'}
    for d in source_step.get('deliveries', []) or []:
        matched = False
        for td in target_step.setdefault('deliveries', []):
            if td.get('order_id') == d.get('order_id') and td.get('sku_id') == d.get('sku_id'):
                td['quantity'] = td.get('quantity', 0) + d.get('quantity', 0)
                matched = True
                break
        if not matched:
            target_step['deliveries'].append(d.copy())
    # unloads: similar to pickups
    for u in source_step.get('unloads', []) or []:
        matched = False
        for tu in target_step.setdefault('unloads', []):
            if tu.get('warehouse_id') == u.get('warehouse_id') and tu.get('sku_id') == u.get('sku_id'):
                tu['quantity'] = tu.get('quantity', 0) + u.get('quantity', 0)
                matched = True
                break
        if not matched:
            target_step['unloads'].append(u.copy())

# --- Trip planning with vehicle-load simulation & guaranteed operations ---
def plan_optimized_trip(env: LogisticsEnvironment,
                        vehicle_obj,
                        orders_with_data: List[Tuple[str, float, float, Dict[str, int], int]],
                        local_inventory: Dict[str, Dict[str, int]],
                        warehouse_nodes: Dict[str, int],
                        max_distance: float) -> Tuple[List[Dict], List[str], float]:
    home_node = warehouse_nodes.get(vehicle_obj.id)
    if home_node is None:
        return [], [], 0.0

    max_w = float(getattr(vehicle_obj, 'capacity_weight', 0.0) or 0.0)
    max_v = float(getattr(vehicle_obj, 'capacity_volume', 0.0) or 0.0)

    chosen_orders: List[str] = []
    pickup_plan: Dict[str, Dict[str, int]] = {}
    total_w = 0.0
    total_v = 0.0

    # select orders greedily
    for order_id, ow, ov, reqs, order_node in orders_with_data:
        if order_id in chosen_orders:
            continue
        if total_w + ow > max_w or total_v + ov > max_v:
            continue
        order_pickups = find_best_warehouses(env, order_id, reqs, local_inventory, order_node)
        if not order_pickups:
            continue

        # quick estimate distance check
        est = 0.0
        for wh_id in order_pickups.keys():
            wh = env.get_warehouse_by_id(wh_id)
            try:
                wh_node = int(wh.location.id) if wh and getattr(wh, 'location', None) else None
            except Exception:
                wh_node = None
            if wh_node is None:
                est += 0.001
            else:
                est += estimate_distance(env, home_node, wh_node)
        est += estimate_distance(env, home_node, order_node)
        if max_distance > 0 and est > max_distance * 0.9:
            continue

        chosen_orders.append(order_id)
        total_w += ow
        total_v += ov
        for wh_id, skus in order_pickups.items():
            pickup_plan.setdefault(wh_id, {})
            for sku_id, q in skus.items():
                pickup_plan[wh_id][sku_id] = pickup_plan[wh_id].get(sku_id, 0) + int(q)
                local_inventory[wh_id][sku_id] = local_inventory[wh_id].get(sku_id, 0) - int(q)

    if not chosen_orders:
        return [], [], 0.0

    # build nodes
    wh_nodes: List[int] = []
    wh_mapping: Dict[int, str] = {}
    for wh_id in pickup_plan.keys():
        wh = env.get_warehouse_by_id(wh_id)
        if not wh or not getattr(wh, 'location', None):
            continue
        try:
            wn = int(wh.location.id)
        except Exception:
            continue
        if wn not in wh_nodes:
            wh_nodes.append(wn)
            wh_mapping[wn] = wh_id

    order_node_map = {oid: on for oid, _, _, _, on in orders_with_data if oid in chosen_orders}
    delivery_nodes = list(order_node_map.values())

    wh_sequence = optimize_delivery_sequence(env, home_node, wh_nodes)
    delivery_sequence = optimize_delivery_sequence(env, home_node, delivery_nodes)

    # planned visit order and connectivity check
    planned_nodes: List[int] = [home_node] + wh_sequence + delivery_sequence + [home_node]
    for i in range(len(planned_nodes) - 1):
        a = planned_nodes[i]
        b = planned_nodes[i + 1]
        sd = segment_distance(env, a, b)
        if sd is None:
            return [], [], 0.0

    # construct steps with simulated load
    steps: List[Dict] = []
    current_node = home_node
    simulated_load: Dict[str, int] = {}

    def load_totals(load_map: Dict[str, int]) -> Tuple[float, float]:
        tw = 0.0
        tv = 0.0
        for sku, q in load_map.items():
            if q <= 0:
                continue
            w, v = get_wv(env, sku)
            tw += float(w) * int(q)
            tv += float(v) * int(q)
        return tw, tv

    # Helper to append intermediate nodes excluding final target (to avoid duplicate target)
    def append_intermediate_nodes_for_op(path: List[int], op_target: int):
        """Append intermediate nodes path[1:-1] (if any). Caller will append the op node separately."""
        if len(path) <= 2:
            return
        for node in path[1:-1]:
            steps.append({'node_id': int(node), 'pickups': [], 'deliveries': [], 'unloads': []})

    # Pickups
    for wn in wh_sequence:
        path_res = a_star_path(env, current_node, wn)
        if not path_res:
            return [], [], 0.0
        path, _ = path_res
        # append intermediate nodes but not the final target yet
        append_intermediate_nodes_for_op(path, wn)

        wh_id = wh_mapping.get(wn)
        pickups_list = []
        for sku_id, q in pickup_plan.get(wh_id, {}).items():
            qty = int(q)
            pickups_list.append({'warehouse_id': wh_id, 'sku_id': sku_id, 'quantity': qty})
            simulated_load[sku_id] = simulated_load.get(sku_id, 0) + qty

        # capacity check
        cur_w, cur_v = load_totals(simulated_load)
        if (max_w > 0 and cur_w > max_w) or (max_v > 0 and cur_v > max_v):
            return [], [], 0.0

        # now append the operation node (the warehouse itself)
        steps.append({'node_id': int(wn), 'pickups': pickups_list, 'deliveries': [], 'unloads': []})
        current_node = wn

    # Deliveries
    for dn in delivery_sequence:
        path_res = a_star_path(env, current_node, dn)
        if not path_res:
            return [], [], 0.0
        path, _ = path_res
        append_intermediate_nodes_for_op(path, dn)

        order_ids = [oid for oid, on in order_node_map.items() if on == dn]
        deliveries = []
        # check simulated load and consume
        for oid in order_ids:
            reqs = env.get_order_requirements(oid) or {}
            for sku, q in reqs.items():
                qreq = int(q)
                have = simulated_load.get(sku, 0)
                if have < qreq:
                    return [], [], 0.0
            # subtract after verifying all SKUs for this order
            for sku, q in reqs.items():
                qreq = int(q)
                simulated_load[sku] = simulated_load.get(sku, 0) - qreq
                if simulated_load[sku] <= 0:
                    del simulated_load[sku]
            for sku, q in reqs.items():
                deliveries.append({'order_id': oid, 'sku_id': sku, 'quantity': int(q)})

        # capacity check (should reduce)
        cur_w, cur_v = load_totals(simulated_load)
        if (max_w > 0 and cur_w > max_w) or (max_v > 0 and cur_v > max_v):
            return [], [], 0.0

        steps.append({'node_id': int(dn), 'pickups': [], 'deliveries': deliveries, 'unloads': []})
        current_node = dn

    # Ensure we've visited all required nodes; append any missing ones and operations.
    visited_nodes = []
    for s in steps:
        nid = int(s['node_id'])
        if not visited_nodes or visited_nodes[-1] != nid:
            visited_nodes.append(nid)
    required_set = set([home_node] + wh_nodes + delivery_nodes + [home_node])
    missing = [n for n in required_set if n not in visited_nodes]

    for target in missing:
        # If it's an op-node (warehouse or delivery) we want to append intermediate nodes then the op node,
        # otherwise append full path nodes (path[1:])
        path_res = a_star_path(env, current_node, target)
        if not path_res:
            return [], [], 0.0
        path, _ = path_res

        # check whether target is a pickup warehouse
        if target in wh_mapping:
            append_intermediate_nodes_for_op(path, target)
            wh_id = wh_mapping[target]
            pickups_list = []
            for sku_id, q in pickup_plan.get(wh_id, {}).items():
                qty = int(q)
                pickups_list.append({'warehouse_id': wh_id, 'sku_id': sku_id, 'quantity': qty})
                simulated_load[sku_id] = simulated_load.get(sku_id, 0) + qty
            cur_w, cur_v = load_totals(simulated_load)
            if (max_w > 0 and cur_w > max_w) or (max_v > 0 and cur_v > max_v):
                return [], [], 0.0
            steps.append({'node_id': int(target), 'pickups': pickups_list, 'deliveries': [], 'unloads': []})
        else:
            # check if target is a delivery node
            order_ids = [oid for oid, on in order_node_map.items() if on == target]
            if order_ids:
                append_intermediate_nodes_for_op(path, target)
                deliveries = []
                for oid in order_ids:
                    reqs = env.get_order_requirements(oid) or {}
                    for sku, q in reqs.items():
                        qreq = int(q)
                        have = simulated_load.get(sku, 0)
                        if have < qreq:
                            return [], [], 0.0
                    for sku, q in reqs.items():
                        qreq = int(q)
                        simulated_load[sku] = simulated_load.get(sku, 0) - qreq
                        if simulated_load.get(sku, 0) <= 0:
                            simulated_load.pop(sku, None)
                        deliveries.append({'order_id': oid, 'sku_id': sku, 'quantity': int(q)})
                cur_w, cur_v = load_totals(simulated_load)
                if (max_w > 0 and cur_w > max_w) or (max_v > 0 and cur_v > max_v):
                    return [], [], 0.0
                steps.append({'node_id': int(target), 'pickups': [], 'deliveries': deliveries, 'unloads': []})
            else:
                # generic visit: append full path movement nodes
                for node in path[1:]:
                    steps.append({'node_id': int(node), 'pickups': [], 'deliveries': [], 'unloads': []})

        current_node = target

    # Return home (append intermediate nodes + home)
    if current_node != home_node:
        path_home_res = a_star_path(env, current_node, home_node)
        if not path_home_res:
            return [], [], 0.0
        path_home, _ = path_home_res
        for node in path_home[1:]:
            steps.append({'node_id': int(node), 'pickups': [], 'deliveries': [], 'unloads': []})

    # final check: compute distance, ensure positive and below limits
    actual_distance = route_distance_using_env(env, steps)
    if actual_distance == float('inf') or actual_distance <= 0.0:
        return [], [], 0.0
    if max_distance > 0 and actual_distance > max_distance:
        return [], [], 0.0

    # Defensive: deduplicate consecutive identical nodes but MERGE their operations
    dedup_steps: List[Dict] = []
    for s in steps:
        if not dedup_steps:
            dedup_steps.append({k: (v.copy() if isinstance(v, list) else v) for k, v in s.items()})
            continue
        last = dedup_steps[-1]
        if int(last['node_id']) == int(s['node_id']):
            # merge operations into last
            _merge_ops(last, s)
        else:
            dedup_steps.append({k: (v.copy() if isinstance(v, list) else v) for k, v in s.items()})

    return dedup_steps, chosen_orders, actual_distance

# --- Solver orchestration ---
def solver(env: LogisticsEnvironment) -> Dict:
    solution = {'routes': []}
    start_time = time.time()

    # Clear caches
    _distance_cache.clear()
    _path_cache.clear()
    _sku_cache.clear()

    all_orders = env.get_all_order_ids() or []
    available_vehicle_ids = env.get_available_vehicles() or []

    # vehicles
    vehicles = []
    for vid in available_vehicle_ids:
        try:
            vehicles.append(env.get_vehicle_by_id(vid))
        except Exception:
            for v in env.get_all_vehicles():
                if getattr(v, 'id', None) == vid:
                    vehicles.append(v)
                    break

    vehicles.sort(key=lambda v: float(getattr(v, 'capacity_weight', 0) or 0) * float(getattr(v, 'capacity_volume', 0) or 0), reverse=True)

    # warehouse nodes per vehicle
    warehouse_nodes: Dict[str, int] = {}
    for v in vehicles:
        home_wh_id = getattr(v, "home_warehouse_id", None)
        home_node = resolve_warehouse_node(env, home_wh_id)
        if home_node is None:
            try:
                home_node = resolve_warehouse_node(env, env.get_vehicle_home_warehouse(v.id))
            except Exception:
                home_node = None
        warehouse_nodes[v.id] = home_node

    # order nodes & local inventory copy
    order_nodes: Dict[str, Optional[int]] = {}
    for order_id in all_orders:
        order_nodes[order_id] = get_order_node(env, order_id)

    local_inventory: Dict[str, Dict[str, int]] = {}
    for wh_id, wh in env.warehouses.items():
        inv = env.get_warehouse_inventory(wh_id) or {}
        local_inventory[wh_id] = {sku: int(q) for sku, q in inv.items()}

    # Precompute order_data
    order_data = []
    for order_id in all_orders:
        ow, ov, reqs = compute_order_weight_volume(env, order_id)
        if ow == float('inf'):
            continue
        on = order_nodes.get(order_id)
        if on is None:
            continue
        order_data.append((order_id, ow, ov, reqs, on))

    order_data.sort(key=lambda x: -(x[1] / max(x[2], 0.01)))

    unassigned = set(oid for oid, _, _, _, _ in order_data)
    vehicle_routes: Dict[str, List[Dict]] = {v.id: [] for v in vehicles}
    vehicle_distances: Dict[str, float] = {v.id: 0.0 for v in vehicles}

    MAX_ITERATIONS = 100
    iteration = 0
    while unassigned and iteration < MAX_ITERATIONS:
        iteration += 1
        progress = False
        for v in vehicles:
            if not unassigned:
                break
            max_dist = float(getattr(v, 'max_distance', 0.0) or 0.0)
            remaining_dist = (max_dist - vehicle_distances[v.id]) if max_dist > 0 else float('inf')
            if remaining_dist <= 0:
                continue

            available_orders = [od for od in order_data if od[0] in unassigned]
            steps, assigned, trip_dist = plan_optimized_trip(env, v, available_orders, local_inventory, warehouse_nodes, remaining_dist)
            if steps and assigned:
                if trip_dist > 0 and trip_dist != float('inf'):
                    vehicle_routes[v.id].extend(steps)
                    vehicle_distances[v.id] += trip_dist
                    for oid in assigned:
                        unassigned.discard(oid)
                    progress = True
        if not progress:
            break

    # Emergency single-order allocation
    if unassigned:
        for oid in list(unassigned):
            od = next((x for x in order_data if x[0] == oid), None)
            if not od:
                continue
            for v in vehicles:
                max_dist = float(getattr(v, 'max_distance', 0.0) or 0.0)
                remaining_dist = (max_dist - vehicle_distances[v.id]) if max_dist > 0 else float('inf')
                steps, assigned, trip_dist = plan_optimized_trip(env, v, [od], local_inventory, warehouse_nodes, remaining_dist)
                if steps and assigned and trip_dist > 0 and trip_dist != float('inf'):
                    vehicle_routes[v.id].extend(steps)
                    vehicle_distances[v.id] += trip_dist
                    unassigned.discard(oid)
                    break

    # Finalize routes and ensure valid non-zero travel
    for v in vehicles:
        r = vehicle_routes.get(v.id, [])
        if not r:
            continue
        home_node = warehouse_nodes.get(v.id)
        if home_node is None:
            continue
        if int(r[0]['node_id']) != home_node:
            r.insert(0, {'node_id': int(home_node), 'pickups': [], 'deliveries': [], 'unloads': []})
        if int(r[-1]['node_id']) != home_node:
            r.append({'node_id': int(home_node), 'pickups': [], 'deliveries': [], 'unloads': []})
        dist = route_distance_using_env(env, r)
        if dist == float('inf') or dist <= 0:
            continue
        solution['routes'].append({'vehicle_id': v.id, 'steps': r})

    elapsed = time.time() - start_time
    fulfilled = len(all_orders) - len(unassigned)
    print("=" * 70)
    print("🚚 MWVRP SOLVER RESULTS (Fixed: operations preserved & load simulated)")
    print("=" * 70)
    print(f"⏱️  Execution Time: {elapsed:.2f}s")
    print(f"📦 Orders Fulfilled: {fulfilled}/{len(all_orders)} ({100 * fulfilled / max(1, len(all_orders)):.1f}%)")
    print(f"🚛 Vehicles Used: {len(solution['routes'])}/{len(vehicles)}")
    if unassigned:
        print(f"⚠️  Unfulfilled Orders ({len(unassigned)}): {list(unassigned)[:10]}")
    for v in vehicles:
        d = vehicle_distances.get(v.id, 0.0)
        if d > 0:
            max_d = float(getattr(v, 'max_distance', 0.0) or 0.0)
            pct = (d / max_d * 100) if max_d > 0 else 0.0
            print(f"   {v.id}: {d:.1f}km / {max_d:.1f}km ({pct:.1f}%)")
    print("=" * 70)

    return solution

if __name__ == "__main__":
    env = LogisticsEnvironment()
    sol = solver(env)
    valid, _, _ = env.validate_solution_complete(sol)
    print("Validation:", valid)
    print("Routes returned:", len(sol.get("routes", [])))
