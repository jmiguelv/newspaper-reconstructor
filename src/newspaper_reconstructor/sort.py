def sort_fragments(fragments: list[dict]) -> list[dict]:
    """Sort fragments by likely continuation based on spatial proximity.

    Uses a Greedy Nearest Neighbor approach:
    Starts with the top-rightmost fragment, and iteratively finds the closest
    unvisited fragment using Euclidean distance between bounding boxes, with
    directional penalties to encourage RTL, top-to-bottom reading flow.
    """
    if not fragments:
        return []

    unvisited = list(fragments)

    # 1. Start with the top-rightmost fragment
    def start_score(f):
        rpos = f.get("hpos", 0) + f.get("width", 0)
        # Quantize rpos to 200px bins so we find the top of the rightmost column
        return (-int(rpos / 200), f.get("vpos", 0))

    current = min(unvisited, key=start_score)
    unvisited.remove(current)

    sorted_fragments = [current]

    while unvisited:
        c_left = current.get("hpos", 0)
        c_right = c_left + current.get("width", 0)
        c_top = current.get("vpos", 0)
        c_bottom = c_top + current.get("height", 0)
        c_cx = c_left + current.get("width", 0) / 2
        c_cy = c_top + current.get("height", 0) / 2

        best_score = float("inf")
        best_frag = None

        for f in unvisited:
            f_left = f.get("hpos", 0)
            f_right = f_left + f.get("width", 0)
            f_top = f.get("vpos", 0)
            f_bottom = f_top + f.get("height", 0)
            f_cx = f_left + f.get("width", 0) / 2
            f_cy = f_top + f.get("height", 0) / 2

            # Shortest Euclidean distance between bounding boxes
            dx_edge = max(0, c_left - f_right, f_left - c_right)
            dy_edge = max(0, c_top - f_bottom, f_top - c_bottom)
            base_dist = (dx_edge**2 + dy_edge**2) ** 0.5

            # Directional vector from current center to candidate center
            dx_center = f_cx - c_cx
            dy_center = f_cy - c_cy

            penalty = 0

            # Jawi is RTL: Moving Right is heavily penalized (unless it's a minor indentation < 100px)
            if dx_center > 100:
                penalty += dx_center * 5

            # Jawi reads Top-to-Bottom: Moving Up is penalized...
            # UNLESS we are also moving significantly Left (which indicates a column jump)
            if dy_center < -50 and dx_center > -100:
                penalty += abs(dy_center) * 5

            # We strongly prefer moving straight down (same column)
            # So if dx_edge is small, it's very cheap. If dx_edge is large, it costs more.
            # We add a multiplier to horizontal distance to prioritize vertical continuation
            score = base_dist + (dx_edge * 2) + penalty

            if score < best_score:
                best_score = score
                best_frag = f

        current = best_frag
        unvisited.remove(current)
        sorted_fragments.append(current)

    return sorted_fragments
