import streamlit as st

nav = st.sidebar.radio("Navigate",["Home",  "Classify"])

if nav == "Home":
    if __name__ == "__main__":
        st.title('Music Genre Classification')
        '''
        ![music.png](https://raw.githubusercontent.com/sarveshspatil111/music-genre-classification/master/music.png)
        '''

if nav == "Classify":
    # Importing required libraries
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Read in track metadata with genre labels
    tracks = pd.read_csv('fma-rock-vs-hiphop.csv')

    # Read in track metrics with the features
    echonest_metrics = pd.read_json('echonest-metrics.json',precise_float=True)

    # Merge the relevant columns of tracks and echonest_metrics
    echo_tracks = echonest_metrics.merge(tracks[['track_id','genre_top']],on='track_id')

    # Inspect the resultant dataframe
    echo_tracks.info()

    

    
    # Define our features 
    features = echo_tracks.drop(['genre_top','track_id'],axis=1)

    # Define our labels
    labels = echo_tracks['genre_top']
    # Import the StandardScaler
    from sklearn.preprocessing import StandardScaler

    # Scale the features and set the values to a new variable
    scaler = StandardScaler()
    scaled_train_features = scaler.fit_transform(features)

    

   

    # Subset only the hip-hop tracks, and then only the rock tracks
    hop_only = echo_tracks.loc[echo_tracks['genre_top'] == 'Hip-Hop']
    rock_only = echo_tracks.loc[echo_tracks['genre_top'] == 'Rock']
    # sample the rocks songs to be the same number as there are hip-hop songs
    rock_only = rock_only.sample(n=len(hop_only),random_state=10)


    # concatenate the dataframes rock_only and hop_only
    rock_hop_bal = pd.concat([hop_only,rock_only])

    # The features, labels, and pca projection are created for the balanced dataframe
    features = rock_hop_bal.drop(['genre_top', 'track_id'], axis=1)
    labels = rock_hop_bal['genre_top']
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import KFold, cross_val_score

    # Set up our K-fold cross-validation
    kf = KFold(n_splits=10)

    rfr = RandomForestClassifier(random_state=10)
        
    rfr.fit(features,labels)

    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials

    def getFeatures(url):
        CLIENT_ID = "a79057ae61b94921b3a987b7158e91a4"
        CLIENT_SECRET = "2f3534b6441f460098c69b3213ae5258"

        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID,
                                                                   client_secret=CLIENT_SECRET))

        # Features = ['acousticness', 'danceability', 'energy', 'instrumentalness',
        #        'liveness', 'speechiness', 'tempo', 'valence']

        results = sp.audio_features(tracks = [url])
        data_json = results[0]
        data = [data_json['acousticness'], data_json['danceability'], data_json['energy'], data_json['instrumentalness'], data_json['liveness'], data_json['speechiness'], data_json['tempo'], data_json['valence']]

        return data

    '''
    # Enter Spotify Track URL
    #### Choose either Hip-Hop or Rock track
    '''
    url = st.text_input("","Paste Here")
    try:
        dt = getFeatures(url)
        res = rfr.predict_proba([dt])
        st.success(f"Hip-Hop : {100 * res[0][0]} %")
        st.success(f"Rock : {100 * res[0][1]} %")
    except Exception:
        pass
