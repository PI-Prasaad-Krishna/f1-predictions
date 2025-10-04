def normalize_list(lst, reverse=False):
    """
    Normalize a list of numbers to [0,1].
    If reverse=True, higher original values become lower normalized values.
    """
    min_val = min(lst)
    max_val = max(lst)
    if max_val == min_val:
        return [1.0 for _ in lst]  # all same values, return 1
    if reverse:
        return [(max_val - x) / (max_val - min_val) for x in lst]
    else:
        return [(x - min_val) / (max_val - min_val) for x in lst]


def predict_gp_podium(drivers_data):
    """
    Predict the podium (top 3) for a GP.

    drivers_data: list of dicts, each with keys:
        - name (str): Driver's name
        - season_points (float): Total season points so far
        - qualifying_pos (int): Qualifying position (1 is best)
        - track_points (float): Points or rating at this track historically
        - team_factor (float): Team competitiveness rating (0-1)

    Returns:
        List of top 3 driver names predicted for podium.
    """

    # Extract raw values for normalization
    season_points = [d['season_points'] for d in drivers_data]
    qualifying_pos = [d['qualifying_pos'] for d in drivers_data]
    track_points = [d['track_points'] for d in drivers_data]
    team_factors = [d['team_factor'] for d in drivers_data]

    # Normalize values
    norm_season = normalize_list(season_points)
    norm_qual = normalize_list(qualifying_pos, reverse=True)  # lower pos better, so reverse
    norm_track = normalize_list(track_points)
    norm_team = normalize_list(team_factors)

    # Weights for each factor
    w_form = 0.4
    w_qual = 0.35
    w_track = 0.2
    w_team = 0.05

    # Calculate scores
    scored_drivers = []
    for i, d in enumerate(drivers_data):
        score = (
            w_form * norm_season[i] +
            w_qual * norm_qual[i] +
            w_track * norm_track[i] +
            w_team * norm_team[i]
        )
        scored_drivers.append({'name': d['name'], 'score': score})

    # Sort descending
    scored_drivers.sort(key=lambda x: x['score'], reverse=True)

    # Return top 3
    return [d['name'] for d in scored_drivers[:3]]


# Example usage
if __name__ == "__main__":
    print("Enter data for drivers. Type 'done' when finished.")

    drivers = []
    while True:
        name = input("Driver name (or 'done'): ").strip()
        if name.lower() == 'done':
            break
        try:
            season_points = float(input(f"Season points for {name}: "))
            qualifying_pos = int(input(f"Qualifying position for {name} (1 is pole): "))
            track_points = float(input(f"Track performance rating for {name} (e.g. avg points at this track): "))
            team_factor = float(input(f"Team factor for {name} (0 to 1 scale): "))
        except ValueError:
            print("Invalid input, try again.")
            continue

        drivers.append({
            'name': name,
            'season_points': season_points,
            'qualifying_pos': qualifying_pos,
            'track_points': track_points,
            'team_factor': team_factor
        })

    if len(drivers) < 3:
        print("Need at least 3 drivers to predict podium.")
    else:
        podium = predict_gp_podium(drivers)
        print("\nPredicted Podium:")
        for i, driver in enumerate(podium, 1):
            print(f"{i}. {driver}")
