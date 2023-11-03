import numpy as np
import pandas as pd
import random

from catboost import CatBoostClassifier
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def jaccard_similarity(x, y):
  """ returns the jaccard similarity between two lists """
  #x = row['title']
  #y = row['cvr_title']
  try:
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)
  except:
    return np.nan

def cosine_sim(text_1, text_2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text_1, text_2])
    return ((tfidf * tfidf.T).A)[0,1]
    
def jaccard_similarity_row(row):
  """ returns the jaccard similarity between two lists """
  x = row['title']
  y = row['cvr_title']
  try:
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)
  except:
    return np.nan    

def cosine_sim_row(row):
    
    text_1 = row['text']
    text_2 = row['cvr_text']
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text_1, text_2])
    return ((tfidf * tfidf.T).A)[0,1]


class CoverFinder:
    """
    A class that aligns similar tracks for a given one and searches the original track among them.

    Attributes
    ----------
    df : str
        path to df used
    model : str
        path to model used
    track_id : str
        track id to search similar tracks for
        

    Methods
    -------
    search_covers() -> List:
        Searches the list of covers.
    show_original() -> dict:
        shows metadata dict for original track.
    show_covers() -> List:
        shows list with metadata dicts for cover tracks.
    show_cover_number() -> int:
        shows number of covers for track.
    """
    def make_data_set(self):
        """creates dataframe with metadata if df_path not given
        """
        lyrics_df = pd.read_json('datasets/lyrics.json', lines=True)
        meta_df = pd.read_json('datasets/meta.json', lines=True)
        meta_df = meta_df.drop(['dttm', 'language', 'genres'], axis=1)
        meta_df = meta_df.dropna()
        meta_df['isrc'] = meta_df['isrc'].str[5:7]
        meta_df = meta_df[meta_df['isrc'] != '5-']
        meta_df['isrc'] = meta_df['isrc'].astype(int) 
        meta_df['title'] = meta_df['title'].astype(str)
        meta_df['isrc'] = np.where(meta_df['isrc'] <= 24, 2000 + meta_df['isrc'], 1900 + meta_df['isrc']) 
        meta_df = lyrics_df.merge(meta_df, how='left', on='track_id').drop('lyricId', axis=1)
        meta_df = meta_df.rename(columns={'isrc': 'release', 'track_id': 'original_track_id'})
        meta_df = meta_df.dropna()
        meta_df['release'] = meta_df['release'].astype(int)
        
        self.df = meta_df
        return 
    
    def train_model(self):
        """trains a model if model_path not given
        """
        meta_df = pd.read_json('datasets/meta.json', lines=True)
        lyrics_df = pd.read_json('datasets/lyrics.json', lines=True)
        covers_df = pd.read_json('datasets/covers.json', lines=True)
        pd.options.mode.chained_assignment = None
        covers_df_1 = covers_df.loc[(covers_df['track_remake_type'] == 'COVER') & (covers_df['original_track_id'].notna())]
        covers_df_1['target'] = 1
        covers_df_1 = covers_df_1.drop('track_remake_type', axis=1)
        cover_ids = covers_df_1['track_id'].to_list()
        original_ids = covers_df_1['original_track_id'].to_list()
        target_1_ids = cover_ids + original_ids
        all_ids = lyrics_df['track_id'].to_list()
        target_0_ids = [id for id in all_ids if id not in target_1_ids]
        target_0_ids = list(set(target_0_ids))
        BATCH = len(target_0_ids) // 2
        selected_target_0_ids = random.sample(list(target_0_ids), BATCH * 2)
        covers_df_0 = pd.DataFrame({'original_track_id': selected_target_0_ids[:BATCH], 'track_id': selected_target_0_ids[BATCH:], 'target': np.zeros(BATCH)})
        covers_for_model = pd.concat([covers_df_1, covers_df_0], ignore_index=True).sample(frac=1).reset_index(drop=True)
        meta_df = meta_df.drop(['dttm', 'language', 'genres'], axis=1)
        meta_df = meta_df.dropna()
        meta_df['isrc'] = meta_df['isrc'].str[5:7]
        meta_df = meta_df[meta_df['isrc'] != '5-']
        meta_df['isrc'] = meta_df['isrc'].astype(int)
        meta_df['title'] = meta_df['title'].astype(str)
        meta_df['isrc'] = np.where(meta_df['isrc'] <= 24, 2000 + meta_df['isrc'], 1900 + meta_df['isrc']) 
        meta_df = lyrics_df.merge(meta_df, how='left', on='track_id').drop('lyricId', axis=1)
        meta_df = meta_df.rename(columns={'isrc': 'release', 'track_id': 'original_track_id'})
        meta_df = meta_df.dropna()
        meta_df['release'] = meta_df['release'].astype(int)
        c_meta_df = meta_df.copy()
        c_meta_df = c_meta_df.rename(columns={'original_track_id': 'track_id', 'title': 'cvr_title', 'release': 'cvr_release', 'duration': 'cvr_duration', 'text': 'cvr_text'})
        covers_meta_df = covers_for_model.merge(meta_df, on='original_track_id', how='inner').merge(c_meta_df, on='track_id', how='inner')
        covers_meta_df = covers_meta_df.drop(['original_track_id', 'track_id'], axis=1)
        covers_meta_df['release_diff'] = abs(covers_meta_df['release'] - covers_meta_df['cvr_release'])
        covers_meta_df = covers_meta_df.drop(['release', 'cvr_release'], axis=1)
        covers_meta_df['duration_diff'] = abs(covers_meta_df['duration'] - covers_meta_df['cvr_duration'])
        covers_meta_df = covers_meta_df.drop(['duration', 'cvr_duration'], axis=1)
        covers_meta_df['title_diff'] = covers_meta_df[['title', 'cvr_title']].apply(jaccard_similarity_row, axis=1)
        covers_meta_df = covers_meta_df.drop(['title', 'cvr_title'], axis=1)
        covers_meta_df['text_diff'] = covers_meta_df.apply(cosine_sim_row, axis=1)
        covers_meta_df = covers_meta_df.drop(['text', 'cvr_text'], axis=1)
        covers_meta_df = covers_meta_df.dropna()
        covers_meta_df['target'] = covers_meta_df['target'].astype(int)
        #covers_meta_df = data_train
        data_train, data_valid = train_test_split(covers_meta_df, stratify=covers_meta_df['target'], test_size=0.25)
        
        features_train = data_train.drop('target', axis=1)
        target_train = data_train['target']
        
        resample = SMOTETomek(tomek=TomekLinks(sampling_strategy='auto'))

        features_samp, target_samp = resample.fit_resample(features_train, target_train)
        
        features_valid = data_valid.drop('target', axis=1)
        target_valid = data_valid['target']
        
        model_1 = CatBoostClassifier(iterations=1758,
                                     learning_rate=0.04120428861906423,
                                     depth=10,
                                     l2_leaf_reg=0.01888103986880093,
                                     bootstrap_type='Bayesian',
                                     random_strength=4.848702461343636,
                                     bagging_temperature=2.5853849579969137,
                                     od_type='IncToDec',
                                     od_wait=11,
                                     verbose=False
                                     )

        model_1.fit(features_samp,
                    target_samp,
                    eval_set=(features_valid, target_valid),
                    plot=False)
        
        self.model = model_1
        return 
    
    def __init__(self, df_path=None, model_path=None):
        """
        Creates an example of CoverFinder class, requires paths to df and model

        Args:
            df_path (str): path to df with metadata
            model_path (str): path to model
        """
        if df_path:
            self.df = pd.read_csv(df_path, sep='\t', encoding='utf-8')
        else:
            self.make_data_set()
        if model_path:
            self.model = CatBoostClassifier().load_model(model_path, format='cbm')
        else:
            self.train_model()
            
        self.search_results = None
        
    def search_covers(self, track_id):
        """looking for similar tracks in df

        Args:
            track_id (str): track_id

        Returns:
            dict: dictionary with similar tracks ids and year of their release {track_id: year}
        """
        try:
        
            track_data = self.df.loc[self.df['original_track_id'] == track_id]
                           
            track_duration = track_data['duration'].values[0]
            track_release = track_data['release'].values[0]
            track_title = track_data['title'].values[0]
            track_text = track_data['text'].values[0]
            
        except:
            print('no data for this track')
            return 'no data for this track'
        
        self.search_results = {}
        self.search_results[track_id] = track_release
        
        for i in self.df.index:
            try:
                compare_data = self.df.iloc[i]
            
                compare_id = compare_data['original_track_id']
                if compare_id == track_id:
                    continue
                compare_duration = compare_data['duration']
                compare_release = compare_data['release']
                compare_title = compare_data['title']
                compare_text = compare_data['text']
                            
                duration_diff = abs(track_duration - compare_duration)
                release_diff = abs(track_release - compare_release)
                title_diff = jaccard_similarity(track_title, compare_title)
                text_diff = cosine_sim(track_text, compare_text)
            
                features = [release_diff, 
                            duration_diff,
                            title_diff,
                            text_diff]
            
                compare_result = self.model.predict(data=features)
                
                if compare_result == 1:
                    self.search_results[compare_id] = compare_release
            except:
                continue
        print('tracks fetched')
        #print(self.search_results)
        return 'tracks fetched'
    
    def show_original(self):
        """
        picks a track with the earliest year of release from search results 

        Returns:
            dict: dictionary with track summary {'id': track_id, 'title': track_title, 'release_year': track_release}
        """
        if self.search_results:
            res = min(self.search_results, key=self.search_results.get)
            
            original_data = self.df.loc[self.df['original_track_id'] == res]
            
            track_release = original_data['release'].values[0]
            track_title = original_data['title'].values[0]
            
            original_dict = {
                'id': res,
                'title': track_title,
                'release_year': track_release
            }
            
            return original_dict
        else:
            print('perform searh first')
            return 'perform searh first'
        
    def show_covers(self):
        """
        returns a list of similar tracks except the original one

        Returns:
            List: list with dictionaries with tracks summaries [ {'id': track_id, 'title': track_title, 'release_year': track_release} ]
        """
        if self.search_results:
            if len(self.search_results) > 1:
                origin_key = min(self.search_results, key=self.search_results.get)
                covers_list = []
                for key in self.search_results:
                    if key != origin_key:
                        cover_data = self.df.loc[self.df['original_track_id'] == key]
            
                        track_release = cover_data['release'].values[0]
                        track_title = cover_data['title'].values[0]
            
                        cover_dict = {
                            'id': key,
                            'title': track_title,
                            'release_year': track_release
                            }
                        
                        covers_list.append(cover_dict)
             
            else:
                return 'no covers'
            
            return covers_list
        else:
            print('perform searh first')
            return 'perform searh first'
        
    def show_cover_number(self):
        """
        returns a number of covers

        Returns:
            int: number of covers
        """
        if self.search_results:
            return len(self.search_results) - 1
        else:
            print('perform search first')
            return 'perform search first'
    
if __name__ == "__main__":
    
    finder = CoverFinder(
        #'datasets/meta_for_model.csv', 'models/cvr_clsfr_mdl.cbm'
        )
    finder.search_covers('deb9b9598176a0bab1212d430b10bd04')
    print(finder.show_covers())
    print(finder.show_cover_number())
    print(finder.show_original())  