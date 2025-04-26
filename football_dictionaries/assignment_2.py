def players_by_position(squads_list):
    pass
def group_players_by_position(squads_data):
    players = players_list_to_dicts(squads_data)
    grouped_by_position = {}

    for player in players:
        position = player['position']
        if position not in grouped_by_position:
            grouped_by_position[position] = []
        grouped_by_position[position].append(player)

    return grouped_by_position
