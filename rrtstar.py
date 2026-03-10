import numpy as np
import math
import random

class Node:
    __slots__ = ['n', 'e', 'parent', 'cost']
    def __init__(self, n, e):
        self.n = n
        self.e = e
        self.parent = None
        self.cost = 0

class RRTStar:
    def __init__(
        self, 
        start, 
        goal, 
        delaunay, 
        weighted_random, 
        potential_fields, 
        step_size=0.25, 
        iter_scale=5, 
        logger=None, 
        check_delaunay=False, 
        safety_margin = 0.4, 
        repulsive_weight = 15.0, 
        goal_region_radius=0.35, 
        search_radius=1.0
        ):
        self.logger = logger
        self.start = Node(start.ned.n, start.ned.e)
        self.goal = Node(goal.ned.n, goal.ned.e)
        self.delaunay = delaunay
        
        if weighted_random is not None and len(weighted_random) > 0:
            s = np.sum(weighted_random)
            if s > 0:
                self.weighted_random = weighted_random / s
            else:
                self.weighted_random = None
        else:
            self.weighted_random = None
        self.step_size = step_size
        self.max_iter = 0
        self.iter_scale = iter_scale
        self.obstacles = potential_fields
        self.node_list = [self.start]
        self.max_nodes_capacity = 8000
        self._node_coords = np.zeros((self.max_nodes_capacity, 2))
        self._node_coords[0] = [start.ned.n, start.ned.e]
        self._node_count = 1 
        
        self.goal_region_radius = goal_region_radius
        self.search_radius = search_radius
        self.path = []
        self.goal_reached = False
        self.safety_margin = safety_margin
        self.best_cost = float('inf')
        self.repulsive_weight = repulsive_weight 
        self.check_delaunay = check_delaunay
        self.best_cost_counter = 0
        
        self.min_node_dist = 0.10
        
        if self.delaunay and self.weighted_random is not None:
            self.active_indices, self.active_weights = self.filter_active_triangles(margin=0.0)
        else:
            self.active_indices = []
            self.active_weights = None
        
        self.start_ignored_obstacles = []
        for obj in self.obstacles:
            dist_to_start = math.hypot(self.start.n - obj.n, self.start.e - obj.e)
            if dist_to_start < (obj.r + self.safety_margin):
                if self.logger:
                    self.logger.warn(f"START W KOLIZJI! Ignoruję przeszkodę ID/Klasa: {getattr(obj, 'object_class', 'unknown')} TYLKO dla pierwszego ruchu.")
                self.start_ignored_obstacles.append(obj)
        
    def plan(self, post_processing=False):
        """Main RRT* planning loop z logiką 'Target Node Count'"""
        min_n = min(self.start.n, self.goal.n)
        max_n = max(self.start.n, self.goal.n)
        min_e = min(self.start.e, self.goal.e)
        max_e = max(self.start.e, self.goal.e)
        
        if max_n-min_n > max_e-min_e:
            margin = (0.5*(max_n-min_n))**(0.8)
            other_len = max_n-min_n+2*margin
            max_n += margin; min_n -= margin
            other_len -= max_e-min_e; other_len /= 2
            margin = other_len
            max_e += margin*1.25; min_e -= margin*1.25
        else:
            margin = (0.5*(max_e-min_e))**(0.8)
            other_len = max_e-min_e+2*margin
            max_e += margin; min_e -= margin
            other_len -= max_n-min_n; other_len /= 2
            margin = other_len
            max_n += margin*1.25; min_n -= margin*1.25
        Area = (max_n-min_n)*(max_e-min_e)
        
        target_node_count = int((Area/(self.step_size**2)) * self.iter_scale)
        target_node_count = min(target_node_count, self.max_nodes_capacity - 1)
        
        max_total_attempts = target_node_count * 4
        
        self.logger.info(f"RRT Start. Target Nodes: {target_node_count}, Area: {Area:.1f}")

        total_attempts = 0
        
        while len(self.node_list) < .9*target_node_count and total_attempts < max_total_attempts:
            
            total_attempts += 1
            
            if total_attempts % 200 == 0:
                if self.logger:
                    self.logger.info(f"Nodes: {len(self.node_list)}/{target_node_count}")

            rand_node = self.get_random_node()
            if rand_node is None: continue

            nearest_node = self.get_nearest_node(self.node_list, rand_node)
            new_node = self.steer(nearest_node, rand_node)

            if self.check_collision_line(nearest_node, new_node):
                neighbors = self.find_neighbors(new_node)
                
                too_close = False
                for node in neighbors:
                    if math.hypot(node.n - new_node.n, node.e - new_node.e) <= self.min_node_dist:
                        too_close = True
                        break
                
                if too_close:
                    continue 

                new_node = self.choose_parent(neighbors, nearest_node, new_node)
                self.node_list.append(new_node)
                
                self._node_coords[self._node_count] = [new_node.n, new_node.e]
                self._node_count += 1
                
                self.rewire(new_node, neighbors)
                    
            if self.reached_goal(new_node):
                if new_node.cost < self.best_cost:
                    prev_cost_str = f"{self.best_cost:.2f}" if self.best_cost != float('inf') else "inf"

                    self.logger.info(
                        f"NEW BEST COST: {new_node.cost:.2f} (prev {prev_cost_str})"
                    )

                    self.best_cost = new_node.cost
                    self.best_cost_counter = 0 
                    self.path = self.generate_final_path(new_node)
                else:
                    if self.best_cost != float('inf'):
                        self.best_cost_counter += 1
            else:
                if self.best_cost != float('inf'):
                    self.best_cost_counter += 1

            if self.best_cost != float('inf') and self.best_cost_counter > target_node_count*0.15 and len(self.node_list) > (target_node_count*0.70):
                self.logger.info(f"RRT Converged. No improvement for {self.best_cost_counter} nodes.")
                break

        if self.best_cost != float('inf') and len(self.path) > 0:
            if post_processing:
                self.logger.info("Running post-processing (Smoothing)...")
                self.smooth_path(min_node_dist=0.6)
        if self.logger:
            self.logger.info(f"RRT Finished. Nodes: {len(self.node_list)}, Total Attempts: {total_attempts}")


    def get_random_node(self):
        """Generate a random node in the map"""
        if self.delaunay and self.active_indices and self.active_weights is not None:
            if random.random() > 0.1:
                chosen_idx_pos = np.random.choice(len(self.active_indices), p=self.active_weights)
                idx = self.active_indices[chosen_idx_pos]
                
                tri = self.delaunay.simplices[idx]
                pts = self.delaunay.points
                
                p0 = pts[tri[0]]; p1 = pts[tri[1]]; p2 = pts[tri[2]]
                
                r1, r2 = random.random(), random.random()
                sq1 = math.sqrt(r1)
                n = (1 - sq1)*p0[0] + (sq1*(1-r2))*p1[0] + (sq1*r2)*p2[0]
                e = (1 - sq1)*p0[1] + (sq1*(1-r2))*p1[1] + (sq1*r2)*p2[1]
                return Node(n, e)
            else:
                return Node(self.goal.n, self.goal.e)
                

        rand_val = random.random()
        
        if rand_val < 0.05: 
            return Node(self.goal.n, self.goal.e)
        elif rand_val < 0.20: 
            t = random.uniform(0.0, 1.0)
            n = self.start.n + t * (self.goal.n - self.start.n)
            e = self.start.e + t * (self.goal.e - self.start.e)
            
            dx = self.goal.n - self.start.n
            dy = self.goal.e - self.start.e
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                perp_x = -dy / length
                perp_y = dx / length
                offset = random.uniform(-2.0, 2.0)
                n += perp_x * offset
                e += perp_y * offset
            return Node(n, e)
        else:
            min_n = min(self.start.n, self.goal.n)
            max_n = max(self.start.n, self.goal.n)
            min_e = min(self.start.e, self.goal.e)
            max_e = max(self.start.e, self.goal.e)
            
            if max_n-min_n > max_e-min_e:
                margin = (0.5*(max_n-min_n))**(0.8)
                other_len = max_n-min_n+2*margin
                max_n += margin; min_n -= margin
                other_len -= max_e-min_e; other_len /= 2
                margin = other_len
                max_e += margin*1.25; min_e -= margin*1.25
            else:
                margin = (0.5*(max_e-min_e))**(0.8)
                other_len = max_e-min_e+2*margin
                max_e += margin; min_e -= margin
                other_len -= max_n-min_n; other_len /= 2
                margin = other_len
                max_n += margin*1.25; min_n -= margin*1.25
            
            n_rand = random.uniform(min_n, max_n)
            e_rand = random.uniform(min_e, max_e)
            return Node(n_rand, e_rand)

    def steer(self, from_node, to_node):
        theta = math.atan2(to_node.e - from_node.e, to_node.n - from_node.n)
        dist = math.hypot(to_node.n - from_node.n, to_node.e - from_node.e)
        
        actual_step = min(self.step_size, dist)
        
        new_node = Node(from_node.n + actual_step * math.cos(theta),
                        from_node.e + actual_step * math.sin(theta))
        
        edge_cost = self.calculate_edge_cost(from_node, new_node)
        new_node.cost = from_node.cost + edge_cost
        
        new_node.parent = from_node
        return new_node

    def check_collision_line(self, node1, node2):
        is_start_node = (abs(node1.n - self.start.n) < 0.0001 and abs(node1.e - self.start.e) < 0.0001)

        if self.check_delaunay and self.delaunay:
            if hasattr(self.delaunay, 'find_simplex'):
                mid_n = (node1.n + node2.n) / 2
                mid_e = (node1.e + node2.e) / 2
                if self.delaunay.find_simplex([mid_n, mid_e]) < 0:
                    return False

        seg_v_n = node2.n - node1.n
        seg_v_e = node2.e - node1.e
        seg_len_sq = seg_v_n**2 + seg_v_e**2
        
        if seg_len_sq <= 0.01:
            return self.is_point_safe(node1.n, node1.e)

        for obj in self.obstacles:
            if obj in self.start_ignored_obstacles:
                if is_start_node:
                    continue
                else:
                    pass

            obs_v_n = obj.n - node1.n
            obs_v_e = obj.e - node1.e
            
            t = (obs_v_n * seg_v_n + obs_v_e * seg_v_e) / seg_len_sq
            t_clamped = max(0.0, min(1.0, t))
            
            closest_n = node1.n + t_clamped * seg_v_n
            closest_e = node1.e + t_clamped * seg_v_e
            
            dist_sq = (closest_n - obj.n)**2 + (closest_e - obj.e)**2
            
            if dist_sq <= (obj.r + self.safety_margin+0.2)**2:
                return False 
                
        return True

    
    def is_point_safe(self, x, y):
        if self.check_delaunay and self.delaunay:
            if hasattr(self.delaunay, 'find_simplex'):
                if self.delaunay.find_simplex([x, y]) < 0:
                    return False
        
        is_start_point = (abs(x - self.start.n) < 0.0001 and abs(y - self.start.e) < 0.0001)

        for obj in self.obstacles:
            if is_start_point and (obj in self.start_ignored_obstacles):
                continue

            dist = (x - obj.n)**2 + (y - obj.e)**2
            if dist <= (obj.r + self.safety_margin)**2:
                return False
        return True

    def find_neighbors(self, new_node):
        active_coords = self._node_coords[:self._node_count]
        
        dists_sq = np.sum((active_coords - np.array([new_node.n, new_node.e]))**2, axis=1)

        radius_sq = self.search_radius ** 2
        neighbor_indices = np.where(dists_sq <= radius_sq)[0]
        
        return [self.node_list[i] for i in neighbor_indices]

    def choose_parent(self, neighbors, nearest_node, new_node):
        min_cost = nearest_node.cost + self.calculate_edge_cost(nearest_node, new_node)
        best_node = nearest_node

        for neighbor in neighbors:
            edge_cost = self.calculate_edge_cost(neighbor, new_node)
            cost = neighbor.cost + edge_cost
            
            if cost < min_cost and self.check_collision_line(neighbor, new_node):
                best_node = neighbor
                min_cost = cost

        new_node.cost = min_cost
        new_node.parent = best_node
        return new_node

    def rewire(self, new_node, neighbors):
        for neighbor in neighbors:
            edge_cost = self.calculate_edge_cost(new_node, neighbor)
            cost = new_node.cost + edge_cost
            
            if cost < neighbor.cost and self.check_collision_line(new_node, neighbor):
                neighbor.parent = new_node
                neighbor.cost = cost

    def reached_goal(self, node):
        """Check if the goal has been reached."""
        return np.linalg.norm([node.n - self.goal.n, node.e - self.goal.e]) < self.goal_region_radius

    def generate_final_path(self, goal_node):
        """Generate the final path from the start to the goal."""
        path = []
        node = goal_node
        while node is not None:
            path.append(node)
            node = node.parent
        return path[::-1] 
    
    def get_nearest_node(self, node_list, rand_node):
        active_coords = self._node_coords[:self._node_count]
        
        dists_sq = np.sum((active_coords - np.array([rand_node.n, rand_node.e]))**2, axis=1)
        min_index = np.argmin(dists_sq)
        return self.node_list[min_index]

    def calculate_edge_cost(self, from_node, to_node):
        seg_v_n = to_node.n - from_node.n
        seg_v_e = to_node.e - from_node.e
        seg_len_sq = seg_v_n*seg_v_n + seg_v_e*seg_v_e
        
        dist = math.sqrt(seg_len_sq)
        
        if dist < 0.001: return 0.0
        
        total_influence = 0.0
        influence_zone = 4.0
        
        is_start_node = (abs(from_node.n - self.start.n) < 0.0001 and abs(from_node.e - self.start.e) < 0.0001)

        for obj in self.obstacles:
            if abs(from_node.n - obj.n) > (dist + influence_zone + obj.r) and \
            abs(from_node.e - obj.e) > (dist + influence_zone + obj.r):
                continue
            
            if is_start_node and (obj in self.start_ignored_obstacles):
                continue

            obs_v_n = obj.n - from_node.n
            obs_v_e = obj.e - from_node.e
            
            t = (obs_v_n * seg_v_n + obs_v_e * seg_v_e) / seg_len_sq if seg_len_sq > 0 else 0
            t_clamped = max(0.0, min(1.0, t))
            
            closest_n = from_node.n + t_clamped * seg_v_n
            closest_e = from_node.e + t_clamped * seg_v_e
            
            dist_to_obj_center = math.hypot(closest_n - obj.n, closest_e - obj.e)
            dist_from_surface = dist_to_obj_center - obj.r
            
            if dist_from_surface < influence_zone:
                if dist_from_surface <= 0.1:
                    penalty = 1000.0
                else:
                    term = (1.0 / dist_from_surface) - (1.0 / influence_zone)
                    penalty = 0.5 * self.repulsive_weight * (term * term)
                total_influence += penalty

        return dist * (1.0 + total_influence)

    def filter_active_triangles(self, margin=2.5):
        valid_indices = []
        valid_weights = []
        
        points = self.delaunay.points
        simplices = self.delaunay.simplices
        
        P1 = np.array([self.start.n, self.start.e])
        P2 = np.array([self.goal.n, self.goal.e])
        
        V_path = P2 - P1
        path_length = np.linalg.norm(V_path)
        
        if path_length < 0.01:
            return list(range(len(simplices))), None

        V_dir = V_path / path_length
        
        for i, simplex in enumerate(simplices):
            p0 = points[simplex[0]]
            p1 = points[simplex[1]]
            p2 = points[simplex[2]]
            
            center_n = (p0[0] + p1[0] + p2[0]) / 3.0
            center_e = (p0[1] + p1[1] + p2[1]) / 3.0
            P_tri = np.array([center_n, center_e])
            
            V_to_tri = P_tri - P1
            projection_dist = np.dot(V_to_tri, V_dir)
            
            if -margin <= projection_dist <= (path_length + margin):
                valid_indices.append(i)
                if self.weighted_random is not None:
                    valid_weights.append(self.weighted_random[i])
                else:
                    valid_weights.append(1.0)
        
        if valid_weights:
            s = np.sum(valid_weights)
            if s > 0:
                valid_weights = np.array(valid_weights) / s
            else:
                valid_weights = None
                
        if self.logger:
            self.logger.info(f"Filtered Triangles: {len(valid_indices)}/{len(simplices)} active. Margin: {margin}m")
            
        return valid_indices, valid_weights
    
    def smooth_path(self, min_node_dist=0.6):
        """
        Wygładza ścieżkę i usuwa punkty leżące zbyt blisko siebie.
        :param min_node_dist: Minimalny wymagany dystans (w metrach) między punktami.
        """
        if not self.path or len(self.path) < 3:
            return

        new_path = [self.path[0]]
        current_idx = 0
        max_lookahead = min(5, len(self.path)) 
        
        while current_idx < len(self.path) - 1:
            found_shortcut = False
            search_end = min(len(self.path) - 1, current_idx + max_lookahead)
            
            for target_idx in range(search_end, current_idx, -1):
                if self.check_collision_line(self.path[current_idx], self.path[target_idx]):
                    new_path.append(self.path[target_idx])
                    current_idx = target_idx
                    found_shortcut = True
                    break
            
            if not found_shortcut:
                current_idx += 1
                new_path.append(self.path[current_idx])

        final_path = [new_path[0]]
        for i in range(1, len(new_path) - 1):
            prev_node = final_path[-1]
            curr_node = new_path[i]
            next_node = new_path[i+1]
            
            dist_to_prev = math.hypot(curr_node.n - prev_node.n, curr_node.e - prev_node.e)
            
            if dist_to_prev < min_node_dist:
                if self.check_collision_line(prev_node, next_node):
                    continue
            
            final_path.append(curr_node)
            
        final_path.append(new_path[-1])
        
        dedup_path = [final_path[0]]
        for node in final_path[1:]:
            if math.hypot(node.n - dedup_path[-1].n, node.e - dedup_path[-1].e) > 0.05:
                dedup_path.append(node)

        if self.logger:
            self.logger.info(
                f"Path smoothing: Zredukowano węzły: {len(self.path)} (Raw) -> "
                f"{len(new_path)} (Raycast) -> {len(dedup_path)} (Odległościowe)"
            )
        
        self.path = dedup_path