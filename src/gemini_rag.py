"""
SASTRA Research Finder - Gemini RAG Module
Fine-tuned for accurate keyword extraction and research analysis.
"""

import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import requests

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Try to import NEW Google GenAI client and types
try:
    from google.genai import Client, types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    Client = None
    types = None


class GeminiRAG:
    """RAG system using Google GenAI (Gemma free-tier models)."""
    
    def __init__(self):
        """Initialize Gemini/Gemma API client."""
        self.api_key = os.getenv('GOOGLE_API_KEY', '')
        self.client = None
        self.model = None
        self.model_name = None
        self._initialized = False
        
        if GEMINI_AVAILABLE and self.api_key:
            try:
                # Initialize Google GenAI client
                # The Client will pick up GOOGLE_API_KEY or GEMINI_API_KEY if set
                self.client = Client(api_key=self.api_key)

                # Fetch models from API
                available_models = self._list_available_models()

                if available_models:
                    print(f"Available models: {', '.join(available_models[:5])}")

                    # PRIORITY LIST (free-tier safe)
                    PREFERRED_MODELS = [
                        "gemma-3-27b-it",
                        "gemma-3-4b-it",
                        "gemma-3n-e4b-it",
                        "gemma-3n-e2b-it",
                        "nano-banana-pro-preview"
                    ]

                    for m in PREFERRED_MODELS:
                        if m in available_models:
                            self.model_name = m
                            self._initialized = True
                            print(f"✓ Gemini API initialized with model: {m}")
                            break
                else:
                    print("Could not fetch models — no compatible model available.")

            except Exception as e:
                print(f"✗ Gemini initialization failed: {e}")
                self._initialized = False
    
    def _list_available_models(self) -> List[str]:
        """Query the Gemini API to get list of available models."""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                models = []

                for model in data.get("models", []):
                    full_name = model.get("name", "")
                    if "/" in full_name:
                        clean = full_name.split("/")[-1]
                    else:
                        clean = full_name

                    if "generateContent" in model.get("supportedGenerationMethods", []):
                        models.append(clean)

                return sorted(models, reverse=True)

            else:
                print(f"Model list API returned {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Failed to fetch model list: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if Gemini/Gemma model is available."""
        return self._initialized and self.model_name is not None
    
    def extract_skills(self, project_title: str, context_keywords: List[str] = None) -> List[str]:
        """Extract technical skills from a project title."""
        if not project_title:
            return []
        
        if self.is_available():
            try:
                return self._extract_skills_gemini(project_title, context_keywords)
            except Exception as e:
                print(f"Gemini skill extraction failed: {e}")
                return self._extract_skills_fallback(project_title, context_keywords)
        else:
            return self._extract_skills_fallback(project_title, context_keywords)
    
    def _extract_skills_gemini(self, project_title: str, context_keywords: List[str] = None) -> List[str]:
        """Use Google GenAI (Gemma) to extract skills."""
        context_str = ""
        if context_keywords:
            context_str = f"\n\nRelated keywords from existing research: {', '.join(context_keywords[:30])}"

        prompt = f"""Extract 8-12 specific technical skills/keywords from this research project title.
Focus on:
- Specific techniques (e.g., "convolutional neural networks" not just "deep learning")
- Domain applications (e.g., "medical image segmentation" not just "segmentation")
- Tools/frameworks if mentioned
- Data types (e.g., "MRI", "CT scan", "time series")

Project Title: "{project_title}"{context_str}

IMPORTANT: Return ONLY a comma-separated list of skills. No explanations, no numbering.
Example output: convolutional neural networks, image segmentation, U-Net architecture, medical imaging, MRI analysis

Skills:"""

        try:
            # Build SDK config object using types.GenerateContentConfig
            # NOTE: we use the types object from google.genai
            cfg = None
            if types is not None:
                cfg = types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=200
                )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=cfg
            )

            if response and getattr(response, "text", None):
                skills_text = response.text.strip()
                # Remove any leading/trailing quotes or colons
                skills_text = re.sub(r'^[:\s"\']+|[:\s"\']+$', '', skills_text)
                
                skills = []
                for skill in skills_text.split(','):
                    skill = skill.strip().lower()
                    # Remove numbering if present
                    skill = re.sub(r'^\d+[\.\)]\s*', '', skill)
                    if skill and len(skill) >= 2 and len(skill) <= 50:
                        skills.append(skill)
                
                return skills[:12]
        except Exception as e:
            print(f"Gemini error: {e}")
        
        return self._extract_skills_fallback(project_title, context_keywords)
    
    def _extract_skills_fallback(self, project_title: str, context_keywords: List[str] = None) -> List[str]:
        """Rule-based skill extraction as fallback."""
        if not project_title:
            return []
        
        title_lower = project_title.lower()
        
        # Common technical patterns to look for
        tech_patterns = [
            # Deep Learning
            r'\b(deep learning|neural network|cnn|rnn|lstm|transformer|bert|gpt|attention mechanism)\b',
            r'\b(convolutional|recurrent|generative|adversarial|autoencoder|encoder-decoder)\b',
            # Machine Learning
            r'\b(machine learning|classification|regression|clustering|supervised|unsupervised|reinforcement)\b',
            r'\b(random forest|svm|support vector|decision tree|naive bayes|xgboost|gradient boosting)\b',
            # NLP
            r'\b(natural language|nlp|text mining|sentiment analysis|named entity|pos tagging|word embedding)\b',
            r'\b(language model|text classification|machine translation|question answering|summarization)\b',
            # Computer Vision
            r'\b(computer vision|image processing|object detection|segmentation|image classification)\b',
            r'\b(yolo|resnet|vgg|inception|efficientnet|u-net|mask rcnn|feature extraction)\b',
            # Data Science
            r'\b(data mining|data analysis|big data|analytics|visualization|feature engineering)\b',
            r'\b(prediction|forecasting|time series|anomaly detection|recommendation system)\b',
            # Domain specific
            r'\b(medical imaging|healthcare|diagnosis|drug discovery|bioinformatics|genomics)\b',
            r'\b(iot|smart city|autonomous|robotics|speech recognition|signal processing)\b',
            r'\b(cybersecurity|intrusion detection|malware|encryption|blockchain|cryptography)\b',
            r'\b(optimization|genetic algorithm|particle swarm|evolutionary|metaheuristic)\b',
            # Data types
            r'\b(mri|ct scan|x-ray|ecg|eeg|ultrasound|satellite imagery)\b',
            r'\b(sensor data|network traffic|social media|financial data|clinical data)\b',
        ]
        
        skills = set()
        
        # Extract pattern matches
        for pattern in tech_patterns:
            matches = re.findall(pattern, title_lower)
            for match in matches:
                if isinstance(match, tuple):
                    skills.update(match)
                else:
                    skills.add(match)
        
        # Also extract multi-word phrases
        words = title_lower.split()
        for i in range(len(words)):
            # Single important words
            if words[i] in {'learning', 'network', 'detection', 'classification', 'segmentation', 
                           'recognition', 'prediction', 'analysis', 'processing'}:
                if i > 0:
                    skills.add(f"{words[i-1]} {words[i]}")
            # Two-word combinations
            if i < len(words) - 1:
                bigram = f"{words[i]} {words[i+1]}"
                if any(kw in bigram for kw in ['based', 'driven', 'using', 'neural', 'deep', 'machine']):
                    skills.add(bigram)
        
        # Add context keywords if provided
        if context_keywords:
            for kw in context_keywords[:10]:
                if len(kw) >= 3:
                    skills.add(kw.lower())
        
        # Filter and clean
        final_skills = []
        for skill in skills:
            skill = skill.strip()
            if len(skill) >= 3 and skill not in {'the', 'and', 'for', 'with', 'using', 'based'}:
                final_skills.append(skill)
        
        return final_skills[:12]
    
    def analyze(self, context: List[Dict[str, Any]], skills: List[str]) -> Dict[str, Any]:
        """
        Generate research analysis using RAG.
        
        Args:
            context: List of publication data with abstracts
            skills: List of skills to analyze
        
        Returns:
            Dictionary with 'analysis' text and 'error' if any
        """
        if not self.is_available():
            return {
                'analysis': None,
                'error': 'Gemini API not configured. Add GOOGLE_API_KEY to .env file.'
            }
        
        if not context:
            return {
                'analysis': None,
                'error': 'No relevant publications found for the given skills.'
            }
        
        try:
            return self._generate_analysis(context, skills)
        except Exception as e:
            return {
                'analysis': None,
                'error': f'Analysis generation failed: {str(e)}'
            }
    
    def _generate_analysis(self, context: List[Dict[str, Any]], skills: List[str]) -> Dict[str, Any]:
        """Generate detailed analysis using Gemini/Gemma."""
        
        # Build context string from publications
        context_parts = []
        for idx, pub in enumerate(context[:15], 1):  # Limit to 15 for context window
            author_id = pub.get('author_id', 'Unknown')
            abstract = pub.get('abstract', '')[:800]  # Limit abstract length
            keywords = ', '.join(pub.get('keywords', [])[:5])
            
            context_parts.append(f"""
[Paper {idx}]
Title: {pub.get('title', 'Untitled')}
Author ID: {author_id}
Year: {pub.get('year', 'N/A')}
Citations: {pub.get('citations', 0)}
Keywords: {keywords}
Abstract: {abstract}
""")
        
        context_text = "\n".join(context_parts)
        skills_text = ", ".join(skills)
        
        prompt = f"""You are a research analyst helping find SASTRA University researchers.

USER'S REQUIRED SKILLS: {skills_text}

RELEVANT PUBLICATIONS FROM SASTRA RESEARCHERS:
{context_text}

Based on these publications, provide a comprehensive analysis in this EXACT format:

## 1. KEY METHODS & TECHNIQUES
- List specific methods, algorithms, and techniques found in the research
- Be specific (e.g., "U-Net for medical image segmentation" not just "deep learning")
- Include at least 5-8 specific techniques

## 2. REPRESENTATIVE PAPERS
For each paper, include the Author ID for reference:
- "Paper Title" (AUTHOR_ID: XXXXX) - Brief description of contribution
- List 5-8 most relevant papers

## 3. REQUIRED TECHNOLOGIES & TOOLS
- Programming languages, frameworks, libraries used
- Specific tools mentioned (TensorFlow, PyTorch, scikit-learn, etc.)
- Hardware requirements if mentioned (GPU, cloud, etc.)

## 4. RECOMMENDED RESEARCHERS
Based on paper count and relevance, suggest:
- Author ID: XXXXX - Brief expertise description
- List 3-5 top researchers by Author ID

## 5. NEXT STEPS FOR DEVELOPERS
- Specific actionable recommendations
- Which papers to read first
- Which researchers to contact

IMPORTANT:
- Always include Author IDs when referencing papers or researchers
- Be specific and technical, not generic
- Base all recommendations on the provided publications only
- Do not make up information not in the context"""

        try:
            cfg = None
            if types is not None:
                cfg = types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=2000
                )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=cfg
            )
            
            if response and getattr(response, "text", None):
                return {
                    'analysis': response.text,
                    'error': None
                }
            else:
                return {
                    'analysis': None,
                    'error': 'Empty response from Gemini API'
                }
        except Exception as e:
            return {
                'analysis': None,
                'error': f'Analysis generation failed: {str(e)}'
            }
    
    def summarize_author(self, profile: Dict[str, Any]) -> str:
        """Generate a brief summary of an author's research focus."""
        if not self.is_available():
            return ""
        
        try:
            # Build context from author's publications
            pubs = profile.get('publications', [])[:10]
            keywords = profile.get('top_keywords', [])[:15]
            
            if not pubs:
                return ""
            
            pub_titles = "\n".join([f"- {p.get('title', '')}" for p in pubs])
            kw_text = ", ".join([k for k, _ in keywords]) if keywords else "Not available"
            
            prompt = f"""Briefly summarize this researcher's focus in 2-3 sentences.

Author: {', '.join(profile.get('name_variants', [])[:2])}
Publications: {profile.get('pub_count', 0)}
Citations: {profile.get('total_citations', 0)}

Top Keywords: {kw_text}

Recent Paper Titles:
{pub_titles}

Summary (2-3 sentences only):"""
            
            cfg = None
            if types is not None:
                cfg = types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=150
                )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=cfg
            )
            
            if response and getattr(response, "text", None):
                return response.text.strip()
        except Exception as e:
            print(f"Author summary failed: {e}")
        
        return ""


# Singleton instance
_rag_instance = None


def get_rag() -> GeminiRAG:
    """Get or create RAG singleton."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = GeminiRAG()
    return _rag_instance
