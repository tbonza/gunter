""" Evaluate your association rules. """


def lift(H, support_data):
    """ 
    Compute lift

    $Lift = \frac{c(A \rightarrow B)}{s(B)}$

    Args:
      H: TODO
      support_data: TODO

    Returns:
      H_scored: H with a lift score appended to each item
    """
    H_scored = []
    for itemset in H:

        X = itemset[0]
        Y = itemset[1]
        XYconf = itemset[2]

        Xs = support_data[X]
        Ys = support_data[Y]

        lift_score = XYconf / (Xs * Ys)

        H_scored.append((X, Y, XYconf, lift_score))

    return H_scored

def interest_factor():
    """ 
    Compute interest factor

    Handles binary variables instead of continuous for lift.

    $I(A,B) = \frac{s(A,B)}{s(A) \times s(B)} = \frac{Nf_{11}}{f_{1+} + f_{+1}}$

    Args:

    Returns:
    
    """
    pass
