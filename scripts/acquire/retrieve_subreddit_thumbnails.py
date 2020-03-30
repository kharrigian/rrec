
###################
### Imports
###################

## Standard Library
import json
## External Imports
import pandas as pd
from tqdm import tqdm
## Local Imports
from rrec.acquire.reddit import RedditData

###################
### Load Metadata
###################

## Load Metadata
metadata = pd.read_csv("./data/raw/metadata/subreddit_metadata.csv", low_memory=False, index_col=0)
names = metadata["name"].tolist()

###################
### Query Image URLs
###################

## Query Data
reddit = RedditData(True)
img_urls = {}
for sub in tqdm(reddit._praw.info(fullnames=names), total=len(names)):
    display_name = sub.display_name
    img_urls[display_name] = {
                "icon_img":sub.icon_img,
                "header_img":sub.header_img,
                "banner_img":sub.banner_img,
                "community_icon":sub.community_icon,

    }

###################
### Format
###################

## Format Into DataFrame
img_urls = pd.DataFrame.from_dict(img_urls, orient="index")

## Format URLS into Dictionaries
img_urls_json = []
for subreddit, row in img_urls.iterrows():
    row_dict = json.loads(row.to_json())
    for x, y in row_dict.items():
        if y is None:
            continue
        if len(y) == 0:
            row_dict[x] = None
    row_dict["subreddit"] = subreddit
    img_urls_json.append(row_dict)

###################
### Cache
###################

## Cache URLS
cache_file = "./data/processed/subreddit_thumbnails.json"
with open(cache_file, "w") as the_file:
    json.dump(img_urls_json, the_file)

