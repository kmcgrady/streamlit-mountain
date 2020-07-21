import streamlit.components.v1 as components

def embed_tweet(status_id):
    components.html("""
        <blockquote id="tweet-{0}">
        </blockquote>
        <script sync src="https://platform.twitter.com/widgets.js"></script>
        <script>
            twttr.widgets.createTweet("{0}", document.getElementById("tweet-{0}"));
        </script>
        """.format(status_id))