def players_as_dictionaries(squads_list):
    pass
def players_as_dictionaries(squads_list):
    keys = ['number', 'position', 'name', 'date_of_birth', 'caps', 'club', 'country', 'club_country', 'year']
    players_list = []

    for player in squads_list:
        player_dict = dict(zip(keys, player))
        players_list.append(player_dict)

    return players_list
