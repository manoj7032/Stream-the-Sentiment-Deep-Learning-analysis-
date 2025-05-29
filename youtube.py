


from googleapiclient.discovery import build


api_key = 'AIzaSyBElcB_NePouOTp_pBZDvJ6Y_a6umwALN0'


youtube = build('youtube', 'v3', developerKey=api_key)

def get_video_comments(video_id):

    comments = []
    next_page_token = None

    while True:
    
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            pageToken=next_page_token,
            maxResults=100, 
            textFormat="plainText"
        )
        response = request.execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        next_page_token = response.get('nextPageToken')

        if not next_page_token:
            break

    return comments


def get_video_data(video_id):
    

    video_response = youtube.videos().list(
        part='snippet',
        id=video_id
    ).execute()

    video_title = video_response['items'][0]['snippet']['title']
    video_thumbnail = video_response['items'][0]['snippet']['thumbnails']['high']['url']

    print(f"Title: {video_title}")
    print(f"Thumbnail: {video_thumbnail}")



    comments = get_video_comments(video_id)

    comments_list=[]

    for i, comment in enumerate(comments, start=1):
    
        comments_list.append(comment)

    return video_title, video_thumbnail, comments_list[:50]


if __name__ == '__main__':
    video_id = 'fsQgc9pCyDU'
    print(get_video_data(video_id))