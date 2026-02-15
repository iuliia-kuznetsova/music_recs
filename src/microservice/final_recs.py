'''
    Final recommendations service

    This microservice is used to generate final recommendations for a given user.
    It uses the offline and online recommendations to generate final recommendations for a given user.

    Usage examples:
    python3 -m src.microservice.final_recs # launch final recommendations service
'''

# ---------- Imports ---------- #
import os
from dotenv import load_dotenv
import requests
from requests.exceptions import ConnectionError
from fastapi import FastAPI

# ---------- Load environment variables ---------- #
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(project_root, '.env'))

offline_recs_service_url= os.getenv('OFFLINE_RECS_SERVICE_URL')
events_service_url = os.getenv('EVENTS_SERVICE_URL')
online_recs_service_url = os.getenv('ONLINE_RECS_SERVICE_URL')

headers = {'Accept': 'application/json'}

# ---------- Initialize an app ---------- #
app = FastAPI(title='recsys_final_recs')

# ---------- Deduplicate ids function ---------- #
def dedup_ids(ids):
    '''
        Removes duplicates from a list.

        Args:
        - ids - list of ids

        Returns:
        - list of ids with duplicates removed
    '''

    # Removing duplicates from the list
    seen = set()
    ids = [id for id in ids if not (id in seen or seen.add(id))]
    # Returning the list with duplicates removed
    return ids

# ---------- Add endpoints for the service ---------- #
# Health check
@app.get('/healthy')
async def healthy():
    '''
        Health check endpoint.

        Returns:
        - dictionary with status 'healthy' or 'unhealthy'
    '''
    # Verifying connection to all services
    try:
        response = requests.get(offline_recs_service_url + '/healthy')
        response = requests.get(events_service_url + '/healthy')
        response = requests.get(online_recs_service_url + '/healthy')
    except ConnectionError:
        return {'status': 'unhealthy'}
    else:
        return {'status': 'healthy'}

# Endpoint: display offline recommendations statistics
@app.get('/offline_recs_stats')
async def stats():
    '''
        Display offline recommendations statistics.

        Returns:
        - dictionary with offline recommendations statistics
    '''
    response = requests.get(offline_recs_service_url + '/stats')
    return {'offline_recs_stats': response.json()}

# Endpoint: get offline recommendations
@app.post('/recommendations_offline')
async def recommendations_offline(user_id: int, k: int = 5):
    '''
        The main service asks the offline recommendation microservice to compute recommendations.
        Displays k offline recommendations.

        Args:
        - user_id - user id to generate recommendations for
        - k - number of recommendations to generate

        Returns:
        - dictionary with recommendations
    '''

    # Sending the request to the offline recommendation microservice
    params = {'user_id': user_id, 'k': k}
    response = requests.post(
        offline_recs_service_url + '/get_recs', params=params, headers=headers
    )
    # Getting the response from the offline recommendation microservice
    response = response.json()

    return {'recs': response}

# Endpoint: display online recommendations
@app.post('/recommendations_online')
async def recommendations_online(user_id: int, k: int = 5, num_events: int = 3):
    '''
        The main service asks the online recommendation microservice to compute recommendations.
        Displays k online recommendations based on last online events.

        Args:
        - user_id - user id to generate recommendations for
        - k - number of recommendations to generate
        - num_events - number of events to consider

        Returns:
        - dictionary with recommendations
    '''

    # Sending the request to the events microservice to get the user's current online events
    params = {'user_id': user_id, 'k': num_events}
    response = requests.post(
        events_service_url + '/get', params=params, headers=headers
    )
    # Getting the response from the events microservice
    events = response.json()
    # Getting the events from the response
    events = events['events']

    # Sending the request to the online recommendation microservice
    tracks = []
    scores = []
    for track_id in events:
        params = {'track_id': track_id, 'k': k}
        response = requests.post(
            online_recs_service_url + '/similar_tracks', headers=headers, params=params
        )
        # Getting the response from the online recommendation microservice
        response = response.json()
        tracks += response['similar_track_id']
        scores += response['similarity_score']
    # Combining the similar tracks and similarity scores
    combined = list(zip(tracks, scores))
    # Sorting the combined list by similarity score in descending order
    combined = sorted(combined, key=lambda x: x[1], reverse=True)
    # Getting the tracks from the combined list
    combined = [track for track, _ in combined]

    # Removing duplicates from recommendations
    combined = dedup_ids(combined)
    # Returning the recommendations
    return {'recs': combined[:k]}

# Endpoint: display final recommendations
@app.post('/recommendations')
async def recommendations(user_id: int, k: int = 50):
    '''
        Find final recommendations based on online and offline recomendations.

        Args:
        - user_id - user id to generate recommendations for
        - k - number of recommendations to generate

        Returns:
        - dictionary with recommendations
    '''

    # Finding both types of recommendations
    result_online = await recommendations_online(user_id=user_id, k=k)
    result_offline = await recommendations_offline(user_id=user_id, k=k)
    
    # Returning the offline recommendations if there is no online history
    if result_online['recs'] == []:
        return {'recs': result_offline['recs']}
    else:
        # Blending online and offline recommendations if online history present
        recs_online = result_online['recs']
        recs_offline = result_offline['recs']

        recs_blended = []
        min_length = min(len(recs_offline), len(recs_online))
        for i in range(min_length):
            # Adding the online recommendations to the even positions of the blended list
            recs_blended.append(recs_online[i])
            # Adding the offline recommendations to the odd positions of the blended list
            recs_blended.append(recs_offline[i])

        # Removing duplicates
        recs_blended = dedup_ids(recs_blended)
        # Returning the blendedrecommendations
        return {'recs': recs_blended[:k]}