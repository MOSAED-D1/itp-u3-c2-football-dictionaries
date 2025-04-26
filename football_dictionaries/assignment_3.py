def players_by_country_and_position(squads_list):
    pass
def group_players_by_country_and_position(squads_data):
    players = players_list_to_dicts(squads_data)
    grouped_by_country = {}

    for player in players:
        country = player['country']
        position = player['position']

        if country not in grouped_by_country:
            grouped_by_country[country] = {}

        if position not in grouped_by_country[country]:
            grouped_by_country[country][position] = []

        grouped_by_country[country][position].append(player)

    return grouped_by_country
