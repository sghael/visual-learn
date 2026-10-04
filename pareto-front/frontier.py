"""Pareto membership for lower cost and higher score, including exact ties."""


def frontier(points):
    """Return every nondominated model id, ordered by cost then descending score.

    A dominating point is no worse on either axis and strictly better on at
    least one. Identical measurements therefore keep both model identities.
    """
    ranked = sorted(points, key=lambda p: (p['cost'], -p['score']))
    out, best_score, best_cost = [], -float('inf'), None
    for point in ranked:
        score, cost = point['score'], point['cost']
        if score > best_score:
            out.append(point['id'])
            best_score, best_cost = score, cost
        elif score == best_score and cost == best_cost:
            out.append(point['id'])
    return out
