# PROJECT IN PROGRESS

# Approaches:
- no_news_no_feature_engineering - sort of a benchmark to compare to, simple model no 
- news_with_feature_engineering - in-place feature calculation by an llm based on news (I want to try 2 apraoches 1. where it is calculated per day and the second ne per time frame)
- news_features_from_rag - let llm ask for news itself and only based on news it picked create features
- news_shaped_based_rag - copy approach from research paper where rag was used to find similarly looking stock spikes, then maybe grab news around that or sth