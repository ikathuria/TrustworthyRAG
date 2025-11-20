from query_complexity import QueryComplexityClassifier
from query_intent import QueryIntentClassifier

qc = QueryComplexityClassifier()
ic = QueryIntentClassifier()

# query = "Compare table data for incidents in 2023 and show a visual chart."
query = input("Enter your query: ")
print("\nQuery:", query)

# Complexity scoring
complexity = qc.classify(query)
print("\nQuery complexity:", complexity)
print("Total difficulty score:", qc.total_score(query))

# Intent classification and routing
intent = ic.classify(query)
weights = ic.get_routing_weights(intent)
print("\nDetected intent:", intent)
print("Routing weights:", weights)
