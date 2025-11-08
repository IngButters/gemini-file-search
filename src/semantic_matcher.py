"""
Semantic Matcher Module

Matches items across different files using semantic understanding.
Uses LLM-based matching for accurate synonym and variation detection.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import pandas as pd
from google.genai.errors import APIError
import json


@dataclass
class MatchResult:
    """Result of a semantic match operation."""
    base_item: str
    base_index: int
    matched_item: Optional[str]
    match_index: int  # -1 if no match
    confidence: str  # 'HIGH', 'MEDIUM', 'LOW', 'NONE'
    similarity_score: float  # 0.0 to 1.0
    reasoning: str = ""


class SemanticMatcher:
    """Matches items using semantic understanding via LLM."""

    def __init__(self, gemini_client):
        """
        Initialize SemanticMatcher.

        Args:
            gemini_client: GeminiChatClient instance for LLM access
        """
        self.gemini_client = gemini_client
        self.match_cache = {}  # Cache for repeated matches

    def find_best_match(self, base_item: str, candidates: List[str],
                       base_index: int = 0) -> MatchResult:
        """
        Find best semantic match for base_item among candidates.

        Args:
            base_item: Item description to match
            candidates: List of candidate item descriptions
            base_index: Index of base item (for tracking)

        Returns:
            MatchResult with best match and confidence
        """
        # Check cache
        cache_key = (base_item, tuple(candidates))
        if cache_key in self.match_cache:
            return self.match_cache[cache_key]

        # If no candidates, return no match
        if not candidates or len(candidates) == 0:
            result = MatchResult(
                base_item=base_item,
                base_index=base_index,
                matched_item=None,
                match_index=-1,
                confidence='NONE',
                similarity_score=0.0,
                reasoning="No candidates available"
            )
            self.match_cache[cache_key] = result
            return result

        # Ask LLM to find best match
        prompt = f"""
Tarea: Encuentra la mejor coincidencia semántica.

Ítem base: "{base_item}"

Candidatos:
{json.dumps(candidates, indent=2, ensure_ascii=False)}

Instrucciones:
- Encuentra qué candidato coincide MEJOR semánticamente con el ítem base
- Considera sinónimos, abreviaciones, variaciones de orden, equivalentes técnicos
- Ejemplos de coincidencias válidas:
  * "LOCALIZACIÓN Y REPLANTEO POR METRO CUADRADO..." ↔ "LOCALIZACIÓN Y REPLANTEO"
  * "DEMOLICIÓN DE MUROS EN MAMPOSTERÍA" ↔ "DEMOLER MURO DE MAMPOSTERÍA"
  * "EXCAVACIÓN MANUAL EN TIERRA" ↔ "EXCAVACIÓN MANUAL EN SUELO"

Responde SOLO con JSON (sin bloques de código markdown):
{{
  "match_index": <número del índice del candidato que mejor coincide, o -1 si ninguno>,
  "confidence": "<HIGH|MEDIUM|LOW|NONE>",
  "similarity_score": <0.0 a 1.0>,
  "reasoning": "<explicación breve en español>"
}}

IMPORTANTE: Responde SOLO con el objeto JSON, sin texto adicional.
"""

        try:
            # Send message to Gemini
            response = self.gemini_client.send_message(prompt)

            if not response:
                raise ValueError("No response from Gemini")

            # Parse JSON response
            response_text = response.text.strip()

            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            match_data = json.loads(response_text)

            # Create MatchResult
            match_index = match_data.get('match_index', -1)
            matched_item = candidates[match_index] if match_index >= 0 and match_index < len(candidates) else None

            result = MatchResult(
                base_item=base_item,
                base_index=base_index,
                matched_item=matched_item,
                match_index=match_index,
                confidence=match_data.get('confidence', 'NONE'),
                similarity_score=match_data.get('similarity_score', 0.0),
                reasoning=match_data.get('reasoning', '')
            )

            # Cache result
            self.match_cache[cache_key] = result
            return result

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Response was: {response.text if response else 'None'}")

            # Return no match on error
            result = MatchResult(
                base_item=base_item,
                base_index=base_index,
                matched_item=None,
                match_index=-1,
                confidence='NONE',
                similarity_score=0.0,
                reasoning=f"Error parsing response: {str(e)}"
            )
            return result

        except APIError as e:
            print(f"API Error during matching: {e}")
            result = MatchResult(
                base_item=base_item,
                base_index=base_index,
                matched_item=None,
                match_index=-1,
                confidence='NONE',
                similarity_score=0.0,
                reasoning=f"API error: {str(e)}"
            )
            return result

    def batch_match(self, base_items: List[str], candidate_items: List[str]) -> List[MatchResult]:
        """
        Match all base items against candidates.

        Args:
            base_items: List of base item descriptions
            candidate_items: List of candidate item descriptions

        Returns:
            List of MatchResult for each base item
        """
        results = []

        print(f"\nMatching {len(base_items)} base items against {len(candidate_items)} candidates...")
        print("This may take a few minutes...\n")

        for i, base_item in enumerate(base_items):
            print(f"Matching item {i+1}/{len(base_items)}: {base_item[:50]}...")

            match_result = self.find_best_match(base_item, candidate_items, base_index=i)
            results.append(match_result)

            # Show match status
            if match_result.confidence in ['HIGH', 'MEDIUM']:
                print(f"  ✓ Matched: {match_result.matched_item[:50]}... ({match_result.confidence})")
            else:
                print(f"  ✗ No match found")

        return results

    def get_match_statistics(self, results: List[MatchResult]) -> Dict:
        """
        Calculate statistics for match results.

        Args:
            results: List of MatchResult objects

        Returns:
            Dictionary with match statistics
        """
        total = len(results)
        high_conf = sum(1 for r in results if r.confidence == 'HIGH')
        medium_conf = sum(1 for r in results if r.confidence == 'MEDIUM')
        low_conf = sum(1 for r in results if r.confidence == 'LOW')
        no_match = sum(1 for r in results if r.confidence == 'NONE')

        matched = high_conf + medium_conf + low_conf

        return {
            'total_items': total,
            'matched': matched,
            'no_match': no_match,
            'match_rate': (matched / total * 100) if total > 0 else 0,
            'high_confidence': high_conf,
            'medium_confidence': medium_conf,
            'low_confidence': low_conf,
            'avg_similarity': sum(r.similarity_score for r in results) / total if total > 0 else 0
        }

    def review_low_confidence_matches(self, results: List[MatchResult],
                                     threshold: str = 'MEDIUM') -> List[MatchResult]:
        """
        Filter matches below confidence threshold for manual review.

        Args:
            results: List of MatchResult objects
            threshold: Confidence threshold ('HIGH', 'MEDIUM', 'LOW')

        Returns:
            List of matches below threshold
        """
        confidence_levels = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'NONE': 0}
        threshold_level = confidence_levels.get(threshold, 2)

        return [r for r in results if confidence_levels.get(r.confidence, 0) < threshold_level]

    def format_match_result(self, result: MatchResult) -> str:
        """
        Format a match result for display.

        Args:
            result: MatchResult to format

        Returns:
            Formatted string
        """
        output = f"\nÍtem Base ({result.base_index + 1}): {result.base_item}\n"

        if result.matched_item:
            output += f"  → Coincidencia: {result.matched_item}\n"
            output += f"  → Confianza: {result.confidence} ({result.similarity_score:.2f})\n"
            if result.reasoning:
                output += f"  → Razonamiento: {result.reasoning}\n"
        else:
            output += f"  → NO ENCONTRADO\n"
            if result.reasoning:
                output += f"  → Razón: {result.reasoning}\n"

        return output
