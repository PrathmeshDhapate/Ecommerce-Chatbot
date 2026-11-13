
import csv
from datetime import datetime
from pathlib import Path

from rag import vector_store, ensure_faq_indexed, RAG_THRESHOLD


# Each case contains:
# (user query, list of FAQ questions considered relevant)

SUPPORTED_CASES = [
    (
        "Can I send a product back?",
        [
            "How do I return an item?",
            "What is your return policy?",
        ],
    ),
    (
        "How do I get a product returned?",
        [
            "How do I return an item?",
            "What is your return policy?",
        ],
    ),
    (
        "How many business days until delivery?",
        [
            "How long does shipping take?",
            "When will my order arrive?",
        ],
    ),
    (
        "When should my package reach me?",
        [
            "How long does shipping take?",
            "When will my order arrive?",
        ],
    ),
    (
        "Do you accept credit cards?",
        [
            "Can I pay with a debit or credit card?",
            "What payment methods do you accept?",
        ],
    ),
    (
        "Can I use my bank card to pay?",
        [
            "Can I pay with a debit or credit card?",
            "What payment methods do you accept?",
        ],
    ),
    (
        "Where can I see my package tracking details?",
        [
            "How can I track my order?",
            "Where can I find my tracking information?",
            "How can I check my order status?",
        ],
    ),
    (
        "How do I find my parcel tracking details?",
        [
            "How can I track my order?",
            "Where can I find my tracking information?",
        ],
    ),
    (
        "I want to stop my order.",
        [
            "Can I cancel my order?",
            "How do I cancel my order?",
        ],
    ),
    (
        "Can I undo an order I placed?",
        [
            "Can I cancel my order?",
            "How do I cancel my order?",
        ],
    ),
    (
        "Can you tell me my order status?",
        [
            "How can I check my order status?",
        ],
    ),
    (
        "Has my parcel already arrived?",
        [
            "Has my order been delivered?",
            "How can I check my order status?",
        ],
    ),
    (
        "How do I know which size to buy?",
        [
            "How do I choose the right size?",
            "Where can I find the size guide?",
        ],
    ),
    (
        "Can you help me choose a size?",
        [
            "How do I choose the right size?",
            "Where can I find the size guide?",
        ],
    ),
    (
        "Is my product covered by a warranty?",
        [
            "How can I check my warranty coverage?",
            "What is the warranty policy?",
        ],
    ),
    (
        "Is my item still under warranty?",
        [
            "How can I check my warranty coverage?",
            "What is the warranty policy?",
        ],
    ),
    (
        "I cannot sign in to my account.",
        [
            "Can you help me with my account?",
            "I can't log in to my account. What should I do?",
        ],
    ),
    (
        "I have trouble accessing my account.",
        [
            "Can you help me with my account?",
            "I can't log in to my account. What should I do?",
        ],
    ),
    (
        "How should I maintain my item?",
        [
            "How should I care for my product?",
        ],
    ),
    (
        "How do I look after my product?",
        [
            "How should I care for my product?",
        ],
    ),
    (
        "What can customer support help me with?",
        [
            "What can you help me with?",
            "What services can you help me with?",
        ],
    ),
    (
        "Tell me about yourself.",
        [
            "Are you a virtual assistant?",
            "Who are you?",
        ],
    ),
    (
        "Hey there!",
        [
            "Good morning",
            "Hi",
        ],
    ),
    (
        "I appreciate your assistance.",
        [
            "Thank you",
            "Thanks for your help",
        ],
    ),
    (
        "I'm leaving now, bye.",
        [
            "Goodbye",
            "See you later",
        ],
    ),
    (
        "What happens if I need to cancel a purchase?",
        [
            "Can I cancel my order?",
            "How do I cancel my order?",
        ],
    ),
    (
        "Where is the sizing chart?",
        [
            "How do I choose the right size?",
            "Where can I find the size guide?",
        ],
    ),
    (
        "Does your loyalty scheme give me any benefits?",
        [
            "What is your loyalty program?",
        ],
    ),
    (
        "I forgot my password and can't get in.",
        [
            "I can't log in to my account. What should I do?",
        ],
    ),
    (
        "How do I earn loyalty rewards?",
        [
            "What is your loyalty program?",
        ],
    ),
]


UNSUPPORTED_CASES = [
    "Do you offer gift wrapping?",
    "What is the weather today?",
    "Can you recommend a movie?",
    "What is the capital of Japan?",
    "Can I get a student discount?",
]


def evaluate_retrieval():
    ensure_faq_indexed()

    output_path = (
        Path(__file__).parent
        / f"retrieval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    csv_rows = []

    total_relevant = 0
    total_retrieved_relevant = 0
    total_accepted_relevant = 0
    total_accepted = 0
    total_top5_results = 0

    print("=" * 70)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 70)

    print(f"FAQ collection count: {vector_store._collection.count()}")
    print(f"RAG threshold: {RAG_THRESHOLD}")
    print(f"Supported test cases: {len(SUPPORTED_CASES)}")
    print(f"Unsupported test cases: {len(UNSUPPORTED_CASES)}")

    # ---------------------------------------------------------
    # Supported queries
    # ---------------------------------------------------------

    for query, expected_faqs in SUPPORTED_CASES:
        expected_set = set(expected_faqs)

        results = vector_store.similarity_search_with_score(
            query,
            k=5,
        )

        retrieved_relevant = 0
        accepted_relevant = 0
        accepted_count = 0

        for rank, (doc, score) in enumerate(results, start=1):
            faq_question = doc.page_content
            accepted = score <= RAG_THRESHOLD
            is_relevant = faq_question in expected_set

            if is_relevant:
                retrieved_relevant += 1

            if accepted:
                accepted_count += 1
                total_accepted += 1

                if is_relevant:
                    accepted_relevant += 1
                    total_accepted_relevant += 1

            csv_rows.append(
                {
                    "query_type": "supported",
                    "query": query,
                    "rank": rank,
                    "retrieved_faq": faq_question,
                    "similarity_score": score,
                    "threshold_accepted": accepted,
                    "expected_relevant": is_relevant,
                    "expected_faqs": " | ".join(expected_faqs),
                }
            )

        total_relevant += len(expected_set)
        total_retrieved_relevant += retrieved_relevant
        total_top5_results += len(results)

        precision_at_5 = (
            retrieved_relevant / len(results)
            if results
            else 0.0
        )

        recall_at_5 = (
            retrieved_relevant / len(expected_set)
            if expected_set
            else 0.0
        )

        accepted_precision = (
            accepted_relevant / accepted_count
            if accepted_count
            else 0.0
        )

        accepted_recall = (
            accepted_relevant / len(expected_set)
            if expected_set
            else 0.0
        )

        print(f"\nQuery: {query}")
        print(f"  Precision@5: {precision_at_5:.3f}")
        print(f"  Recall@5: {recall_at_5:.3f}")
        print(f"  Accepted FAQs: {accepted_count}")
        print(f"  Accepted precision: {accepted_precision:.3f}")
        print(f"  Accepted recall: {accepted_recall:.3f}")

    # ---------------------------------------------------------
    # Unsupported queries
    # ---------------------------------------------------------

    unsupported_passes = 0
    total_unsupported_accepted = 0

    for query in UNSUPPORTED_CASES:
        results = vector_store.similarity_search_with_score(
            query,
            k=5,
        )

        accepted_results = 0

        for rank, (doc, score) in enumerate(results, start=1):
            accepted = score <= RAG_THRESHOLD

            if accepted:
                accepted_results += 1
                total_unsupported_accepted += 1

            csv_rows.append(
                {
                    "query_type": "unsupported",
                    "query": query,
                    "rank": rank,
                    "retrieved_faq": doc.page_content,
                    "similarity_score": score,
                    "threshold_accepted": accepted,
                    "expected_relevant": False,
                    "expected_faqs": "",
                }
            )

        passed = accepted_results == 0

        if passed:
            unsupported_passes += 1

        print(f"\nUnsupported query: {query}")
        print(f"  Accepted FAQs: {accepted_results}")
        print(f"  Result: {'PASS' if passed else 'FAIL'}")

    # ---------------------------------------------------------
    # Aggregate metrics
    # ---------------------------------------------------------

    precision_at_5 = (
        total_retrieved_relevant / total_top5_results
        if total_top5_results
        else 0.0
    )

    recall_at_5 = (
        total_retrieved_relevant / total_relevant
        if total_relevant
        else 0.0
    )

    accepted_precision = (
        total_accepted_relevant / total_accepted
        if total_accepted
        else 0.0
    )

    accepted_recall = (
        total_accepted_relevant / total_relevant
        if total_relevant
        else 0.0
    )

    # ---------------------------------------------------------
    # Export CSV
    # ---------------------------------------------------------

    fieldnames = [
        "query_type",
        "query",
        "rank",
        "retrieved_faq",
        "similarity_score",
        "threshold_accepted",
        "expected_relevant",
        "expected_faqs",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    # ---------------------------------------------------------
    # Final report
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL RETRIEVAL EVALUATION SUMMARY")
    print("=" * 70)

    print(f"Precision@5:       {precision_at_5:.3f}")
    print(f"Recall@5:          {recall_at_5:.3f}")
    print(f"Accepted precision:{accepted_precision:.3f}")
    print(f"Accepted recall:   {accepted_recall:.3f}")

    print(
        f"Unsupported rejected: "
        f"{unsupported_passes}/{len(UNSUPPORTED_CASES)}"
    )

    print(
        f"Accepted FAQs across unsupported queries: "
        f"{total_unsupported_accepted}"
    )

    print(f"\nCSV saved to: {output_path}")
    print(f"Exported retrieval rows: {len(csv_rows)}")


if __name__ == "__main__":
    evaluate_retrieval()
