# Apriori Algorithm Implementation
from itertools import combinations

def generate_candidates(itemsets, length):
    """ Generate candidate itemsets of a given length """
    return set([item1.union(item2) for item1 in itemsets for item2 in itemsets if len(item1.union(item2)) == length])


def filter_candidates(transactions, candidates, min_support):
    """ Filter candidate itemsets by minimum support """
    itemset_counts = {candidate: 0 for candidate in candidates}
    for transaction in transactions:
        for candidate in candidates:
            if candidate.issubset(transaction):
                itemset_counts[candidate] += 1
    return {itemset: count for itemset, count in itemset_counts.items() if count >= min_support}


def apriori(transactions, min_support):
    """ Apriori algorithm for frequent itemset generation """
    # Generate 1-itemsets
    itemsets = set(frozenset([item]) for transaction in transactions for item in transaction)
    frequent_itemsets = filter_candidates(transactions, itemsets, min_support)
    result = dict(frequent_itemsets)

    k = 2
    while frequent_itemsets:
        # Generate candidates of size k
        candidates = generate_candidates(frequent_itemsets.keys(), k)
        # Filter candidates
        frequent_itemsets = filter_candidates(transactions, candidates, min_support)
        result.update(frequent_itemsets)
        k += 1
    return result


def association_rules(frequent_itemsets, min_confidence):
    """ Generate association rules from frequent itemsets """
    rules = []
    for itemset in frequent_itemsets:
        for length in range(1, len(itemset)):
            for subset in combinations(itemset, length):
                antecedent = frozenset(subset)
                consequent = itemset - antecedent
                support = frequent_itemsets[itemset]
                confidence = support / frequent_itemsets[antecedent] if frequent_itemsets.get(antecedent) else 0
                if confidence >= min_confidence:
                    rules.append((antecedent, consequent, confidence))
    return rules


# Example usage
transactions = [
    {'milk', 'bread', 'butter'},
    {'beer', 'bread'},
    {'milk', 'bread', 'beer', 'butter'},
    {'beer', 'butter'},
    {'bread', 'butter'}
]
min_support = 2
min_confidence = 0.6

frequent_itemsets = apriori(transactions, min_support)
rules = association_rules(frequent_itemsets, min_confidence)

print("Frequent Itemsets:", frequent_itemsets)
print("Association Rules:")
for antecedent, consequent, confidence in rules:
    print(f"{set(antecedent)} -> {set(consequent)} (confidence: {confidence:.2f})")
