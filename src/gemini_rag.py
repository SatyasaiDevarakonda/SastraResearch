"""
SASTRA Research Finder - Gemini RAG Module
Compatible with google-generativeai (OLD SDK)
Updated to support Streamlit Cloud secrets.
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

# Try to import OLD Google GenerativeAI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None


class GeminiRAG:
    """RAG system using Google GenerativeAI (Gemini models)."""
    
    def __init__(self):
        """Initialize Gemini API client."""
        self.api_key = self._get_api_key()
        self.model = None
        self.model_name = None
        self._initialized = False
        
        if GEMINI_AVAILABLE and self.api_key:
            try:
                # Configure the OLD SDK
                genai.configure(api_key=self.api_key)
                
                # Try to list available models
                available_models = self._list_available_models()
                
                if available_models:
                    print(f"✓ Found {len(available_models)} available models")
                    
                    # PRIORITY LIST for OLD SDK (use Gemini models)
                    PREFERRED_MODELS = [
                        "gemini-1.5-flash",
                        "gemini-1.5-flash-8b",
                        "gemini-1.0-pro",
                        "gemini-pro"
                    ]
                    
                    for model_name in PREFERRED_MODELS:
                        if model_name in available_models:
                            self.model_name = model_name
                            self.model = genai.GenerativeModel(model_name)
                            self._initialized = True
                            print(f"✓ Gemini API initialized with model: {model_name}")
                            break
                    
                    if not self._initialized:
                        # Fallback to first available model
                        self.model_name = available_models[0]
                        self.model = genai.GenerativeModel(self.model_name)
                        self._initialized = True
                        print(f"✓ Gemini API initialized with model: {self.model_name}")
                else:
                    print("✗ No models available")
                    
            except Exception as e:
                print(f"✗ Gemini initialization failed: {e}")
                self._initialized = False
        else:
            if not GEMINI_AVAILABLE:
                print("✗ google-generativeai library not installed")
            if not self.api_key:
                print("✗ No API key found")
    
    def _get_api_key(self) -> str:
        """
        Get API key from multiple sources in priority order:
        1. Streamlit secrets (for cloud deployment)
        2. Environment variable
        3. .env file
        """
        # Priority 1: Streamlit secrets (for cloud deployment)
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'GOOGLE_API_KEY' in st.secrets:
                print("✓ API key loaded from Streamlit secrets")
                return st.secrets['GOOGLE_API_KEY']
        except (ImportError, FileNotFoundError, KeyError, AttributeError):
            pass
        
        # Priority 2: Environment variable
        api_key = os.getenv('GOOGLE_API_KEY', '')
        if api_key:
            print("✓ API key loaded from environment variable")
            return api_key
        
        # Priority 3: Try loading .env file explicitly
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)
            api_key = os.getenv('GOOGLE_API_KEY', '')
            if api_key:
                print("✓ API key loaded from .env file")
                return api_key
        except ImportError:
            pass
        
        print("✗ No API key found in any location")
        return ''
    
    def _list_available_models(self) -> List[str]:
        """Get list of available models using OLD SDK."""
        try:
            models = genai.list_models()
            model_names = []
            
            for model in models:
                # Extract model name (e.g., "models/gemini-pro" -> "gemini-pro")
                name = model.name.split('/')[-1] if '/' in model.name else model.name
                
                # Check if model supports generateContent
                if 'generateContent' in [method for method in model.supported_generation_methods]:
                    model_names.append(name)
            
            return model_names
        except Exception as e:
            print(f"✗ Failed to list models: {e}")
            # Return default models as fallback
            return ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-pro"]
    
    def is_available(self) -> bool:
        """Check if Gemini model is available."""
        return self._initialized and self.model is not None
    
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
        """Use Gemini to extract skills."""
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
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=200
                )
            )
            
            if response and response.text:
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
                'error': 'Gemini API not configured. Add GOOGLE_API_KEY to Streamlit secrets or .env file.'
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
        """Generate detailed analysis using Gemini."""
        
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
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=2000
                )
            )
            
            if response and response.text:
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
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=150
                )
            )
            
            if response and response.text:
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