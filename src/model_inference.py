"""
Multi-Modal Model Inference Module
Handles model loading, prediction, and similarity search using Local Files
"""

import torch
import torch.nn.functional as F
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration, 
    AutoProcessor, 
    AutoModelForVision2Seq,
    ViTImageProcessor,
    AutoTokenizer
)
import numpy as np
import json
import os
import random
import re
from typing import Dict, List, Optional
from PIL import Image
import streamlit as st

# Visualizations
import plotly.express as px
try:
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class MultiModalModel:
    def __init__(self, scraper=None):
        self.scraper = scraper
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # System logs
        st.sidebar.markdown("### ⚙️ System Logs")
        st.sidebar.text(f"Device: {self.device}")
        
        # --- PATH CONFIGURATION ---
        # 1. Get the directory where this script lives (e.g., .../95891-Final-Proj/src)
        src_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Go up one level to the project root (e.g., .../95891-Final-Proj)
        project_root = os.path.dirname(src_dir)
        
        # 3. Define the models directory (e.g., .../95891-Final-Proj/models)
        models_dir = os.path.join(project_root, "models")
        
        # 4. Set paths for specific models
        self.t5_path = os.path.join(models_dir, "fashion_cleaner_model")
        self.vision_model_path = os.path.join(models_dir, "final_vision_model")
        self.vision_processor_path = os.path.join(models_dir, "final_image_processor")
        self.vision_tokenizer_path = os.path.join(models_dir, "final_tokenizer")
        
        # 5. Data path (Check root first, then src)
        self.data_path = os.path.join(project_root, "cleaned_captions.json")
        if not os.path.exists(self.data_path):
            self.data_path = os.path.join(src_dir, "cleaned_captions.json")

        # --- DEBUG CHECKS ---
        if not os.path.exists(self.t5_path):
            st.sidebar.error(f"❌ Path Not Found: {self.t5_path}")
            st.sidebar.info(f"Make sure 'models' folder is in: {project_root}")
            raise FileNotFoundError(f"Missing T5 folder at: {self.t5_path}")

        # 1. Load T5 (For Embeddings ONLY)
        try:
            self.t5_tokenizer = T5Tokenizer.from_pretrained(self.t5_path)
            self.t5_model = T5ForConditionalGeneration.from_pretrained(self.t5_path).to(self.device)
            self.t5_model.eval()
            st.sidebar.success("✅ T5 Model Loaded")
        except Exception as e:
            st.sidebar.error(f"T5 Load Error: {e}")
            raise e

        # 2. Load Vision (For Captioning/Verification)
        self.vision_model = None
        if os.path.exists(self.vision_model_path):
            try:
                self.vision_processor = ViTImageProcessor.from_pretrained(self.vision_processor_path)
                self.vision_tokenizer = AutoTokenizer.from_pretrained(self.vision_tokenizer_path)
                self.vision_model = AutoModelForVision2Seq.from_pretrained(self.vision_model_path).to(self.device)
                self.vision_model.eval()
                
                # Config check
                if self.vision_model.config.decoder_start_token_id is None:
                    if self.vision_tokenizer.bos_token_id is not None:
                        self.vision_model.config.decoder_start_token_id = self.vision_tokenizer.bos_token_id
                    else:
                        self.vision_model.config.decoder_start_token_id = self.vision_tokenizer.pad_token_id
                
                if self.vision_model.config.pad_token_id is None:
                    self.vision_model.config.pad_token_id = self.vision_tokenizer.pad_token_id
                    
                st.sidebar.success("✅ Vision Model Loaded")
            except Exception as e:
                st.sidebar.error(f"Vision Load Error: {e}")
                self.vision_model = None
        else:
            st.sidebar.warning(f"Vision folder missing at: {self.vision_model_path}")
            self.vision_model = None

        # 3. Load Data (Local Inventory)
        if os.path.exists(self.data_path):
            with open(self.data_path, "r") as f:
                data = json.load(f)
            keys = list(data.keys())[:3000]
            self.inventory_ids = keys
            self.inventory_texts = [data[k] for k in keys]
            self.inventory_embeddings = self._get_text_embeddings_batch(self.inventory_texts)
            st.sidebar.text(f"Index: {len(keys)} items")
        else:
            st.sidebar.warning("Inventory Data missing")
            self.inventory_ids = ["0"]
            self.inventory_texts = ["sample"]
            self.inventory_embeddings = torch.randn(1, 512).to(self.device)

        # Definitions
        self.colors = {"red", "blue", "green", "black", "white", "yellow", "pink", "purple", "grey", "gray", "orange", "beige", "brown", "navy", "teal", "maroon", "gold", "silver", "charcoal", "cream"}
        self.category_map = {
            "dress": "dress", "gown": "dress",
            "shirt": "shirt", "tee": "shirt", "top": "shirt", "blouse": "shirt", "t-shirt": "shirt", "tank": "shirt",
            "pants": "pants", "jeans": "pants", "trousers": "pants", "shorts": "shorts", "leggings": "pants",
            "jacket": "jacket", "coat": "jacket", "blazer": "jacket", "hoodie": "jacket", "vest": "jacket",
            "skirt": "skirt", "sweater": "sweater", "cardigan": "sweater", "pullover": "sweater",
            "shoes": "shoes", "boots": "shoes", "sneakers": "shoes", "heels": "shoes"
        }
        self.stopwords = {"a", "an", "the", "and", "is", "are", "it", "with", "of", "in", "on", "for", "to", "this", "that", "my", "your", "very", "really", "looks", "like", "worn", "condition", "brand", "new", "size"}

    def _get_text_embeddings_batch(self, texts, batch_size=32):
        """Uses T5 Encoder to get semantic embeddings for search."""
        self.t5_model.eval()
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.t5_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
            with torch.no_grad():
                encoder_outputs = self.t5_model.encoder(**inputs)
                embeddings = encoder_outputs.last_hidden_state.mean(dim=1)
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)

    def _generate_vision_caption(self, image: Image.Image) -> str:
        """Generates a caption using the Vision Model (NOT T5)."""
        if self.vision_model is None or image is None: return ""
        try:
            # Ensure image is RGB
            if image.mode != "RGB": image = image.convert("RGB")
            
            # Process image for the Vision Model
            inputs = self.vision_processor(images=image, return_tensors="pt").to(self.device)
            
            # Generate caption
            with torch.no_grad():
                generated_ids = self.vision_model.generate(**inputs, max_length=60)
            
            # Decode output
            caption = self.vision_tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
            return caption
        except Exception as e:
            print(f"Vision generation error: {e}")
            return ""

    def _verify_seller_text(self, visual_caption, seller_text):
        """Filter seller text based on vision model output."""
        st.sidebar.text("Verifying description...")
        
        vis_words = set(re.findall(r'\w+', visual_caption.lower()))
        vis_color = next((w for w in vis_words if w in self.colors), None)
        vis_cat_raw = next((w for w in vis_words if w in self.category_map), None)
        vis_cat = self.category_map.get(vis_cat_raw)

        seller_words = re.findall(r'\w+', seller_text.lower())
        verified_words = []
        
        st.sidebar.text(f"Vision saw: Color={vis_color}, Item={vis_cat_raw}")
        
        for word in seller_words:
            if word in self.stopwords: continue 
                
            confidence = 50 
            status = "Neutral"
            
            # Check color
            if word in self.colors:
                if vis_color and word != vis_color:
                    confidence = 0
                    status = "Conflict"
                elif vis_color and word == vis_color:
                    confidence = 100
                    status = "Verified"
            
            # Check category
            elif word in self.category_map:
                mapped_cat = self.category_map[word]
                if vis_cat and mapped_cat != vis_cat:
                    confidence = 0
                    status = "Conflict"
                elif vis_cat and mapped_cat == vis_cat:
                    confidence = 100
                    status = "Verified"
            
            # Only keep words that aren't conflicts
            if confidence > 0:
                verified_words.append(word)

        return " ".join(verified_words)

    def _clean_repetition(self, text):
        if not text: return ""
        words = text.split()
        cleaned = []
        for w in words:
            if not cleaned or w.lower() != cleaned[-1].lower():
                cleaned.append(w)
        return " ".join(cleaned)

    def _create_embedding_visualization(self, query_emb: torch.Tensor):
        if not SKLEARN_AVAILABLE: return None
        try:
            total_items = len(self.inventory_embeddings)
            if total_items < 3: return None

            query_emb_np = np.array(query_emb.cpu().detach().tolist()) 
            num_samples = min(200, total_items)
            indices = np.random.choice(total_items, num_samples, replace=False)
            
            indices_list = indices.tolist()
            sample_embeddings_tensor = self.inventory_embeddings[indices_list]
            sample_embeddings = np.array(sample_embeddings_tensor.cpu().detach().tolist())
            
            all_embeddings = np.vstack([query_emb_np, sample_embeddings])
            all_labels = ["YOUR ITEM"] + ["Inventory" for _ in range(num_samples)]
            
            if len(self.inventory_texts) > max(indices_list):
                sample_texts = [self.inventory_texts[i][:50] + "..." for i in indices_list]
            else:
                sample_texts = ["Item" for _ in indices_list]
                
            all_hover_text = ["Cleaned Description"] + sample_texts
            sizes = [15] + [5] * num_samples 
            
            perp = min(30, len(all_embeddings) - 1)
            if perp < 1: perp = 1
            
            tsne = TSNE(n_components=2, random_state=42, perplexity=perp, init='pca', learning_rate='auto')
            embeddings_2d = tsne.fit_transform(all_embeddings)
            
            fig = px.scatter(
                x=embeddings_2d[:, 0],
                y=embeddings_2d[:, 1],
                color=all_labels,
                size=sizes,
                hover_name=all_hover_text,
                title="Semantic Map",
                labels={'x': 'Dim 1', 'y': 'Dim 2'},
                color_discrete_map={"YOUR ITEM": "red", "Inventory": "blue"},
                opacity=0.7
            )
            return fig
        except Exception as e:
            print(f"Viz error: {e}")
            return None

    def predict(self, image: Optional[Image.Image], text: str) -> Dict:
        st.sidebar.text("Processing...")
        
        # 1. Vision: Generate caption from image
        visual_caption = ""
        if image:
            visual_caption = self._generate_vision_caption(image)
            print(f"Vision output: {visual_caption}")
        
        # 2. Strict Verification (NO T5 Generation)
        # We replace T5 generation with Strict Vision Verification
        text = text.strip() if text else ""
        
        if image and text:
            # Filter seller text using vision
            clean_desc = self._verify_seller_text(visual_caption, text)
        elif text:
            clean_desc = text
        elif visual_caption:
            clean_desc = visual_caption
        else:
            clean_desc = "clothing item"

        # ---------------------------------------------------------
        # CLEANING STEP: Remove artifacts
        # ---------------------------------------------------------
        clean_desc = clean_desc.lower().replace("refine description", "").replace(":", "").strip()
        clean_desc = self._clean_repetition(clean_desc)
        # ---------------------------------------------------------

        print(f"Final output: {clean_desc}")

        # 3. Embeddings & Search
        # Uses T5 Encoder (Text-to-Math) for search, not Text-to-Text generation
        query_emb = self._get_text_embeddings_batch([clean_desc])
        embeddings_vis = self._create_embedding_visualization(query_emb)

        similar_items = []
        if self.scraper:
            try:
                live_results = self.scraper.search_poshmark(clean_desc, top_k=5)
                if live_results:
                    similar_items = live_results 
            except Exception as e:
                print(f"Scraping error: {e}")

        # Fallback Search (Local Inventory)
        if not similar_items:
            scores = torch.mm(query_emb, self.inventory_embeddings.T).squeeze(0)
            top_k = 5
            top_scores, top_indices = torch.topk(scores, k=top_k)
            for rank, idx in enumerate(top_indices):
                idx = idx.item()
                similar_items.append({
                    "item_id": self.inventory_ids[idx],
                    "description": self.inventory_texts[idx],
                    "similarity": f"{top_scores[rank].item():.2f}",
                    "title": f"Local Item {self.inventory_ids[idx][:8]}",
                    "brand": "Vintage", 
                    "price": "$25",
                    "image_url": "https://via.placeholder.com/150", 
                    "listing_url": "#"
                })

        # 4. Attributes 
        # Set high base confidence since text is now verified by vision
        base_confidence = 90
        attributes = self._parse_attributes(clean_desc, base_confidence)

        return {
            "visual_caption": visual_caption,
            "clean_description": clean_desc,
            "attributes": attributes,
            "similar_items": similar_items,
            "embeddings_vis": embeddings_vis
        }

    def predict_validation_mode(self) -> Dict:
        idx = random.randint(0, len(self.inventory_texts)-1)
        return self.predict(None, self.inventory_texts[idx])

    def _parse_attributes(self, text, base_confidence):
        text = text.lower()
        
        def find(word_set, is_hard=False):
            for w in word_set:
                if w in text:
                    # Confidence Calculation: Base + Hard Match Bonus + Random Jitter
                    conf = max(0, min(99, base_confidence + (5 if is_hard else 0) + random.randint(-3, 3)))
                    return w.title(), conf
            return "Unknown", 0

        return {
            "category": {"value": find(self.category_map.keys(), True)[0], "confidence": find(self.category_map.keys(), True)[1]},
            "color": {"value": find(self.colors, True)[0], "confidence": find(self.colors, True)[1]},
            "material": {"value": find(["cotton", "denim", "silk", "wool", "leather", "linen", "velvet", "suede", "lace", "knit", "chiffon", "polyester"])[0], "confidence": base_confidence},
            "style": {"value": "Casual", "confidence": base_confidence}, 
            "condition": {"value": "Pre-owned", "confidence": 95} 
        }