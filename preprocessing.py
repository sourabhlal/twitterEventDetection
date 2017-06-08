import  string, re, time

def stem(token):
	"""
	Stems token
	:param token: token to be stemmed
	Returns token: stemmed token
	"""
	if token.endswith("ing"):
		token = token[:-3]
	elif token.endswith("ed"):
		token = token[:-2]
	elif token.endswith("es"):
		token = token[:-2]
	elif token.endswith("s") and len(token) > 3 and token[-2] in "wrtpsdfgklmnbvcz":
		token = token[:-1]
	return token

def remove_symbol_headTail(token):
	"""
	Removes symbols from the head and tail of a token (example: "***@^&!%happy!!!!!!!!!" --> "happy")
	:param token: token to be un-padded
	Returns: un-padded token
	"""
	f_index = 0
	for x in token:
		if x not in string.punctuation:
			f_index = token.index(x)
			break
	b_index = len(token)
	for i,x in enumerate(reversed(token)):
		if x not in string.punctuation:
			b_index = i
			break
	if b_index == 0:
		return token[f_index:]
	else:
		return token[f_index:-b_index]

def removeEmojis(text):
	"""
	Removes emojis from text
	:param text: string with emojis
	Returns: string without emojis
	"""
	return ''.join(c for c in text if c <= '\uFFFF')

def get_tokens(tweetText,removeStopwords):
	"""
	Takes a string and tokenizes it
	:param tweetText: string to be tokenized
	:param removeStopwords: boolean flag to specify whether stopwords should be removed or not
	Returns words: list of tokens
	"""

	#remove urls, hashtags, @s from tweets
	tweetText = re.sub(r'http\S+', '', tweetText)
	tweetText = re.sub(r'@\S+', '', tweetText)
	tweetText = re.sub(r'#\S+', '', tweetText)
	tweetText = removeEmojis(tweetText)
	tweetText = tweetText.lower()

	# clean the tweets and split only the words
	words = bytes(tweetText,'utf-8').decode('utf-8')\
		.translate(string.punctuation).split()

	# stem the words & remove stopwords
	stopwords = ["i","a","about","an","and","are","as","at","be","by","com","for","from","how","in","is","it","of","on","or","that","this","to","was","what","when","where","who","will","with","the","www", "you", "me", "so", "my","they","your","but","i'm","he","his","if","do","it's","we","him","her","has"]
	if removeStopwords:
		words = [word for word in words if word not in stopwords]
	filtered_words = [word for word in words if len(word)>1]
	filtered_words = [remove_symbol_headTail(w) for w in filtered_words]
	processed_words = [stem(w) for w in filtered_words]

	# remove any empty strings
	words = [word for word in words if word not in [" ", ""]]

	# remove duplicates
	words = list(set(processed_words))
	return words