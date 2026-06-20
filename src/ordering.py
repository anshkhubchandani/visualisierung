import pandas as pd


def compute_similarity_matrix(df, feature_columns):
    """
    Compute similarity between all axes/features.

    We use absolute Pearson correlation:
    high value  = features behave similarly
    low value   = features behave differently
    """

    features = df[feature_columns]

    similarity_matrix = features.corr().abs()

    return similarity_matrix



def greedy_axis_order(similarity_matrix, mode="SIM"):
    """
    Create an axis order using a simple greedy algorithm.

    SIM:
        choose the most similar next axis

    DIS:
        choose the most dissimilar next axis
    """

    remaining_axes = list(similarity_matrix.columns)

    # start with first axis
    current_axis = remaining_axes.pop(0)

    order = [current_axis]
    print (order)

    while remaining_axes:

        if mode == "SIM":
            next_axis = max(
                remaining_axes,
                key=lambda axis: similarity_matrix.loc[current_axis, axis]
            )
            print("SIM: next axis is " + next_axis)
        elif mode == "DIS":
            next_axis = min(
                remaining_axes,
                key=lambda axis: similarity_matrix.loc[current_axis, axis]
            )
        else:
            raise ValueError("mode must be 'SIM' or 'DIS'")

        order.append(next_axis)
        print(next_axis + " added to the order")
        remaining_axes.remove(next_axis)
        current_axis = next_axis

    return order


if __name__ == "__main__":
  if __name__ == "__main__":
    from preprocessing import load_and_preprocess

    df = load_and_preprocess()
    feature_columns = df.drop(columns=["class"]).columns

    sim_matrix = compute_similarity_matrix(df, feature_columns)

    sim_order = greedy_axis_order(sim_matrix, mode="SIM")
    dis_order = greedy_axis_order(sim_matrix, mode="DIS")

    print("\nSIM order:")
    print(sim_order)

    print("\nDIS order:")
    print(dis_order)
    
    #sim_matrix = compute_similarity_matrix(df, feature_columns)
    # #print(sim_matrix)

def print_axis_order_debug(similarity_matrix, order, name):
    print(f"\n{name} axis order:")
    print(order)

    print(f"\n{name} neighboring similarities:")

    values = []

    for left, right in zip(order[:-1], order[1:]):
        similarity = similarity_matrix.loc[left, right]
        values.append(similarity)
        print(f"{left:25s} -> {right:25s}: {similarity:.3f}")

    print(f"\n{name} average neighboring similarity: {sum(values) / len(values):.3f}")