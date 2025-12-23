#!/usr/bin/env python3
"""
Parse simple sentences into (Subject, Predicate, Object) triples.

Uses spaCy for dependency parsing to extract SPO structure.
"""

import json
import sys
from pathlib import Path
from typing import Tuple, Optional, List
import spacy


class SPOParser:
    """Extract Subject-Predicate-Object triples from simple sentences."""

    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model 'en_core_web_sm'...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            self.nlp = spacy.load("en_core_web_sm")

    def parse(self, sentence: str) -> Optional[Tuple[str, str, str]]:
        """
        Parse sentence into (subject, predicate, object) triple.

        Returns:
            (subject, predicate, object) or None if parse fails
        """
        doc = self.nlp(sentence)

        # Strategy 1: Find subject via dependency parsing
        subject = None
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass"):
                # Get full noun phrase
                subject = self._get_noun_phrase(token)
                break

        if not subject:
            # Fallback: First noun phrase
            noun_chunks = list(doc.noun_chunks)
            if noun_chunks:
                subject = noun_chunks[0].text
            else:
                return None

        # Strategy 2: Find main verb and build predicate
        predicate_tokens = []
        main_verb = None

        for token in doc:
            if token.pos_ == "VERB" and token.dep_ in ("ROOT", "ccomp"):
                main_verb = token
                break

        if not main_verb:
            return None

        # Build predicate from verb + particles + prepositions
        predicate_tokens.append(main_verb.text)

        # Add auxiliaries (is, has, etc)
        for child in main_verb.children:
            if child.dep_ == "aux":
                predicate_tokens.insert(0, child.text)

        # Add particles (sitting ON, looking AT)
        for child in main_verb.children:
            if child.dep_ in ("prep", "prt", "agent"):
                predicate_tokens.append(child.text)

        predicate = " ".join(predicate_tokens)

        # Strategy 3: Find object
        object_ = None

        # Look for direct object
        for token in doc:
            if token.dep_ in ("dobj", "pobj", "attr"):
                object_ = self._get_noun_phrase(token)
                break

        if not object_:
            # Fallback: Last noun phrase (if different from subject)
            noun_chunks = list(doc.noun_chunks)
            if len(noun_chunks) > 1:
                object_ = noun_chunks[-1].text
            elif len(noun_chunks) == 1 and noun_chunks[0].text != subject:
                object_ = noun_chunks[0].text

        if not object_:
            return None

        # Clean up
        subject = subject.strip()
        predicate = predicate.strip()
        object_ = object_.strip()

        return (subject, predicate, object_)

    def _get_noun_phrase(self, token) -> str:
        """Get full noun phrase containing this token."""
        # Find the noun phrase this token belongs to
        for chunk in token.doc.noun_chunks:
            if token in chunk:
                return chunk.text

        # Fallback: just the token + its children
        words = [token.text]
        for child in token.children:
            if child.dep_ in ("det", "amod", "compound"):
                words.insert(0, child.text)

        return " ".join(words)


def parse_observations(input_path: Path, output_path: Path):
    """
    Parse observations JSONL into SPO triples.

    Input: JSONL with {image, sentences}
    Output: JSONL with {image, sentence, subject, predicate, object}
    """
    parser = SPOParser()

    triples = []
    total_sentences = 0
    parsed_count = 0

    with input_path.open() as f:
        for line in f:
            obs = json.loads(line)

            for sentence in obs['sentences']:
                total_sentences += 1

                result = parser.parse(sentence)

                if result:
                    subject, predicate, object_ = result
                    triples.append({
                        'image': obs['image'],
                        'sentence': sentence,
                        'subject': subject,
                        'predicate': predicate,
                        'object': object_
                    })
                    parsed_count += 1
                else:
                    print(f"Failed to parse: {sentence}")

    # Write output
    with output_path.open('w') as f:
        for triple in triples:
            f.write(json.dumps(triple) + '\n')

    print(f"Parsed {parsed_count}/{total_sentences} sentences into SPO triples")
    print(f"Triples saved to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Parse sentences into SPO triples')
    parser.add_argument('--input', type=Path, required=True, help='Input observations JSONL')
    parser.add_argument('--output', type=Path, default=Path('triples.jsonl'), help='Output triples JSONL')

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    parse_observations(args.input, args.output)


if __name__ == '__main__':
    main()
