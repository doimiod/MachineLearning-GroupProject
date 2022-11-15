import tweepy
 
# API keyws that yous saved earlier
api_key = "K1YzLpWNN4FqwW8s1oEb2w9hr"
api_secrets = "xbdDA1xDnUkipmeJ7CevCKhtwUrw7QEE6yYjYjvrPjPmw6TUS2"
access_token = "1439151678985580546-doTMyvE7Qa6lgPcJpdlMwrS4JzZYay"
access_secret = "SJ4NOQqnnTYtfIB94Au6CXxAN1OiBGSsi9wvOfEewdCBv"
 
# Authenticate to Twitter
auth = tweepy.OAuthHandler(api_key,api_secrets)
auth.set_access_token(access_token,access_secret)
 
api = tweepy.API(auth)
 
try:
    api.verify_credentials()
    print('Successful Authentication')
except:
    print('Failed authentication')