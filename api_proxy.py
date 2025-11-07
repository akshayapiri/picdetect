#!/usr/bin/env python3
"""
Backend API proxy server for PicDetect
This server proxies image classification requests to Hugging Face API
to avoid CORS issues.

Usage:
    python3 api_proxy.py

The server will run on http://localhost:8001
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import os
from pathlib import Path
from PIL import Image

# Configure a local cache directory for Hugging Face downloads to avoid permission issues
BASE_DIR = Path(__file__).parent
_CACHE_DIR = BASE_DIR / '.cache' / 'huggingface'
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('TRANSFORMERS_CACHE', str(_CACHE_DIR))
os.environ.setdefault('HF_HOME', str(_CACHE_DIR))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

try:
    from transformers import pipeline
    _TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TRANSFORMERS_AVAILABLE = False

# Candidate labels for zero-shot classification (animals, objects, foods, etc.)
LABEL_CANDIDATES = sorted({
    # Animals
    'animal', 'cat', 'dog', 'bird', 'horse', 'cow', 'pig', 'sheep', 'chicken', 'duck',
    'rabbit', 'elephant', 'lion', 'tiger', 'bear', 'wolf', 'fox', 'deer', 'squirrel',
    'mouse', 'rat', 'hamster', 'turtle', 'snake', 'lizard', 'frog', 'fish', 'shark',
    'dolphin', 'whale', 'octopus', 'butterfly', 'bee', 'spider', 'ant', 'sloth',
    'penguin', 'polar bear', 'zebra', 'giraffe', 'monkey', 'panda', 'koala', 'kangaroo',
    'hedgehog', 'raccoon', 'skunk', 'badger', 'owl', 'eagle', 'hawk', 'parrot', 'goat',
    'lobster', 'crab', 'seal', 'otter',
    # Vehicles
    'vehicle', 'car', 'truck', 'bus', 'bicycle', 'bike', 'motorcycle', 'airplane', 'train',
    'boat', 'ship', 'van', 'suv', 'taxi', 'scooter', 'helicopter', 'jet', 'subway', 'tram',
    'tractor', 'ambulance', 'fire truck', 'rocket', 'spaceship',
    # Electronics & appliances
    'phone', 'smartphone', 'tablet', 'laptop', 'computer', 'television', 'camera', 'speaker',
    'headphones', 'microphone', 'keyboard', 'mouse device', 'game console', 'drone',
    'washing machine', 'refrigerator', 'microwave', 'air conditioner', 'fan',
    # Household objects & furniture
    'book', 'book jacket', 'bookshelf', 'magazine', 'comic book', 'notebook', 'textbook',
    'pen', 'pencil', 'paintbrush', 'scissors', 'ruler', 'calculator', 'lamp', 'light bulb',
    'chair', 'stool', 'sofa', 'couch', 'table', 'desk', 'bed', 'cabinet', 'dresser', 'shelf',
    'mirror', 'clock', 'watch', 'backpack', 'suitcase', 'umbrella', 'bucket', 'bottle', 'cup',
    'mug', 'plate', 'bowl', 'spoon', 'fork', 'knife', 'toothbrush', 'towel', 'pillow', 'blanket',
    'basket', 'laundry basket', 'trash can', 'vacuum cleaner', 'broom', 'mop', 'door', 'window',
    'curtain', 'plant pot', 'flower vase',
    # Clothing & accessories
    'clothing', 'shirt', 't-shirt', 'jacket', 'coat', 'dress', 'skirt', 'jeans', 'shorts',
    'sweater', 'hoodie', 'suit', 'tie', 'scarf', 'gloves', 'hat', 'cap', 'helmet', 'shoes',
    'sneakers', 'boots', 'sandals', 'socks', 'belt', 'bag', 'wallet', 'watch accessory',
    # Food & drink
    'food', 'apple', 'banana', 'orange', 'grape', 'strawberry', 'watermelon', 'pineapple',
    'pizza', 'burger', 'sandwich', 'hot dog', 'fries', 'salad', 'soup', 'steak', 'fish dish',
    'sushi', 'rice', 'noodles', 'pasta', 'cake', 'cupcake', 'cookie', 'ice cream', 'donut',
    'bread', 'bagel', 'croissant', 'cheese', 'egg', 'chocolate', 'coffee', 'tea', 'juice',
    'soda', 'water bottle', 'wine glass',
    # Plants & nature
    'plant', 'tree', 'flower', 'rose', 'sunflower', 'cactus', 'grass', 'leaf', 'forest',
    'mountain', 'river', 'lake', 'ocean', 'beach', 'desert', 'snow', 'cloud', 'sky', 'sunset',
    'sunrise', 'rainbow', 'volcano', 'waterfall', 'rock', 'stone',
    # Sports & leisure
    'ball', 'football', 'basketball', 'soccer ball', 'baseball bat', 'tennis racket', 'golf club',
    'skateboard', 'surfboard', 'snowboard', 'bicycle helmet', 'yoga mat', 'dumbbell', 'treadmill',
    # Musical instruments
    'guitar', 'piano', 'violin', 'drum', 'trumpet', 'saxophone', 'flute', 'clarinet', 'oboe',
    'microphone',
    # Buildings & places
    'house', 'building', 'apartment', 'castle', 'palace', 'temple', 'church', 'bridge', 'tower',
    'skyscraper', 'stadium', 'school', 'classroom', 'kitchen', 'bedroom', 'bathroom', 'office',
    'library', 'bookstore', 'laboratory', 'factory', 'warehouse',
    # Tools & equipment
    'tool', 'hammer', 'screwdriver', 'wrench', 'drill', 'saw', 'knife tool', 'pliers', 'axe',
    'shovel', 'rake', 'ladder', 'tape measure', 'toolbox',
    # Toys & games
    'toy', 'doll', 'teddy bear', 'lego', 'puzzle', 'board game', 'playing card', 'kite',
    'balloon',
    # Misc
    'art', 'painting', 'sculpture', 'camera tripod', 'fireplace', 'gift', 'present', 'flag',
    'traffic light', 'stop sign', 'street sign', 'parking meter', 'bench park', 'fountain',
    'shopping cart', 'shopping bag', 'baby stroller', 'bicycle basket'
})

# Lazily initialized Hugging Face pipelines (downloaded on first use)
_clip_classifier = None
_image_classifier = None

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        data = request.get_json()
        
        # Get image data (can be base64 string)
        image_data = data.get('image') or data.get('image_base64')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 if needed
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Convert base64 to bytes
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            return jsonify({'error': f'Invalid base64 image: {str(e)}'}), 400
        
        if not _TRANSFORMERS_AVAILABLE:
            return jsonify({
                'error': 'Missing dependency: transformers. Install with "pip install transformers torch pillow".'
            }), 500

        global _clip_classifier, _image_classifier
        if _clip_classifier is None:
            try:
                _clip_classifier = pipeline(
                    task='zero-shot-image-classification',
                    model='openai/clip-vit-base-patch32'
                )
                print('✅ Loaded CLIP zero-shot image classifier: openai/clip-vit-base-patch32')
            except Exception as model_error:  # pragma: no cover
                print(f'⚠️  Failed to load CLIP classifier: {model_error}')
                _clip_classifier = None
        if _image_classifier is None:
            try:
                _image_classifier = pipeline(
                    task='image-classification',
                    model='google/vit-base-patch16-224'
                )
                print('✅ Loaded transformers image-classification pipeline: google/vit-base-patch16-224')
            except Exception as model_error:  # pragma: no cover
                print(f'⚠️  Failed to load ViT classifier: {model_error}')
                _image_classifier = None

        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as image_error:
            return jsonify({
                'error': 'Invalid image data',
                'details': str(image_error)
            }), 400
        clip_predictions = []
        if _clip_classifier is not None:
            try:
                clip_predictions = _clip_classifier(
                    pil_image,
                    candidate_labels=list(LABEL_CANDIDATES),
                    hypothesis_template='a photo of {}'
                )
            except Exception as clip_error:
                print(f'⚠️  CLIP classifier failed: {clip_error}')
                clip_predictions = []

        if clip_predictions:
            top_prediction = clip_predictions[0]
            label = top_prediction.get('label', 'Unknown')
            score = float(top_prediction.get('score', 0.0))

            result = {
                'name': format_label(label),
                'category': categorize_label(label),
                'confidence': score,
                'description': generate_description(label),
                'source': 'clip-zero-shot',
                'alternatives': [
                    {
                        'name': format_label(p.get('label', 'Unknown')),
                        'confidence': float(p.get('score', 0.0)),
                        'category': categorize_label(p.get('label', 'Unknown'))
                    }
                    for p in clip_predictions[:5]
                ]
            }

            return jsonify(result)

        if _image_classifier is not None:
            try:
                predictions = _image_classifier(pil_image, top_k=5)
            except Exception as inference_error:
                print(f'⚠️  ViT classifier inference failed: {inference_error}')
                predictions = []

            if predictions:
                top_prediction = predictions[0]
                label = top_prediction.get('label', 'Unknown')
                score = float(top_prediction.get('score', 0.0))

                result = {
                    'name': format_label(label),
                    'category': categorize_label(label),
                    'confidence': score,
                    'description': generate_description(label),
                    'source': 'transformers-vit',
                    'alternatives': [
                        {
                            'name': format_label(p.get('label', 'Unknown')),
                            'confidence': float(p.get('score', 0.0)),
                            'category': categorize_label(p.get('label', 'Unknown'))
                        }
                        for p in predictions[:5]
                    ]
                }

                return jsonify(result)

        print('⚠️  All model pipelines failed; using heuristic fallback')
        return use_fallback_classification(image_bytes, reason='model_failure')
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def format_label(label):
    """Convert labels like 'tabby, tabby cat' to readable format"""
    return label.replace('_', ' ').split(',')[0].title()

def categorize_label(label):
    """Categorize the label - expanded to handle many more categories"""
    label_lower = label.lower()
    
    # Animals - comprehensive list
    animals = ['cat', 'dog', 'bird', 'horse', 'cow', 'pig', 'sheep', 'chicken', 
               'duck', 'rabbit', 'elephant', 'lion', 'tiger', 'bear', 'wolf',
               'fox', 'deer', 'squirrel', 'mouse', 'rat', 'hamster', 'turtle',
               'snake', 'lizard', 'frog', 'fish', 'shark', 'dolphin', 'whale',
               'octopus', 'butterfly', 'bee', 'spider', 'ant', 'sloth', 'penguin',
               'zebra', 'giraffe', 'monkey', 'ape', 'gorilla', 'panda', 'koala',
               'kangaroo', 'hedgehog', 'raccoon', 'skunk', 'badger']
    
    if any(animal in label_lower for animal in animals):
        return 'Animal'
    
    # Vehicles - comprehensive list
    vehicles = ['car', 'truck', 'bus', 'bicycle', 'bike', 'motorcycle', 
                'airplane', 'train', 'boat', 'ship', 'van', 'suv', 'taxi',
                'scooter', 'helicopter', 'jet', 'subway', 'tram',
                'ferry', 'yacht', 'cruise', 'tractor', 'ambulance', 'fire truck']
    if any(vehicle in label_lower for vehicle in vehicles):
        return 'Vehicle'
    
    # Food - comprehensive list
    food = ['apple', 'banana', 'orange', 'bread', 'pizza', 'burger', 'sandwich',
            'cake', 'cookie', 'ice cream', 'coffee', 'tea', 'milk', 'cheese',
            'meat', 'chicken', 'beef', 'pork', 'fish', 'rice', 'pasta', 'noodle',
            'soup', 'salad', 'vegetable', 'fruit', 'berry', 'grape', 'strawberry']
    if any(food_item in label_lower for food_item in food):
        return 'Food'
    
    # Plants - comprehensive list
    plants = ['flower', 'rose', 'tree', 'plant', 'leaf', 'grass', 'bush', 'shrub',
              'fern', 'cactus', 'mushroom', 'herb', 'vegetable', 'garden']
    if any(plant in label_lower for plant in plants):
        return 'Plant'
    
    # Electronics
    electronics = ['computer', 'laptop', 'phone', 'smartphone', 'tablet', 'television',
                  'tv', 'camera', 'monitor', 'keyboard', 'mouse', 'speaker', 'headphone',
                  'microphone', 'radio', 'remote', 'charger', 'battery']
    if any(electronic in label_lower for electronic in electronics):
        return 'Electronic'
    
    # Furniture
    furniture = ['chair', 'table', 'desk', 'sofa', 'couch', 'bed', 'cabinet', 'shelf',
                 'wardrobe', 'dresser', 'stool', 'bench', 'ottoman']
    if any(item in label_lower for item in furniture):
        return 'Furniture'
    
    # Clothing
    clothing = ['shirt', 'pants', 'dress', 'jacket', 'coat', 'hat', 'cap', 'shoe',
                'sneaker', 'boot', 'sock', 'glove', 'scarf', 'tie', 'belt']
    if any(item in label_lower for item in clothing):
        return 'Clothing'
    
    # Buildings/Architecture
    buildings = ['house', 'building', 'tower', 'skyscraper', 'church', 'temple',
                 'bridge', 'monument', 'statue', 'castle', 'palace']
    if any(building in label_lower for building in buildings):
        return 'Architecture'
    
    # Nature/Outdoor
    nature = ['mountain', 'hill', 'valley', 'river', 'lake', 'ocean', 'beach',
              'forest', 'jungle', 'desert', 'snow', 'ice', 'cloud', 'sunset', 'sunrise']
    if any(item in label_lower for item in nature):
        return 'Nature'
    
    return 'Object'

def generate_description(label):
    """Generate a description based on the label - expanded for many categories"""
    label_lower = label.lower()
    formatted_name = format_label(label)
    
    # Animals
    if 'cat' in label_lower:
        return f'A {formatted_name} - a beloved domestic pet known for its independent nature, agility, and affectionate behavior.'
    elif 'dog' in label_lower:
        return f'A {formatted_name} - a loyal companion and one of humanity\'s oldest friends, known for intelligence and devotion.'
    elif 'bird' in label_lower:
        return f'A {formatted_name} - a feathered creature capable of flight, known for its beautiful songs and diverse species.'
    elif 'horse' in label_lower:
        return f'A {formatted_name} - a majestic animal known for its strength, speed, and long history with humans.'
    elif 'fish' in label_lower or 'shark' in label_lower:
        return f'A {formatted_name} - an aquatic creature that lives in water, known for its diverse species and adaptations.'
    
    # Food
    elif 'apple' in label_lower or 'fruit' in label_lower:
        return f'A {formatted_name} - a nutritious food item that provides vitamins and energy.'
    elif 'pizza' in label_lower or 'burger' in label_lower:
        return f'A {formatted_name} - a popular food item enjoyed by people around the world.'
    
    # Vehicles
    elif 'car' in label_lower or 'vehicle' in label_lower:
        return f'A {formatted_name} - a motorized vehicle designed for transportation on roads.'
    elif 'bicycle' in label_lower or 'bike' in label_lower:
        return f'A {formatted_name} - a human-powered vehicle with two wheels, great for exercise and transportation.'
    
    # Plants
    elif 'flower' in label_lower or 'rose' in label_lower:
        return f'A {formatted_name} - a beautiful flowering plant that adds color and fragrance to gardens and bouquets.'
    elif 'tree' in label_lower:
        return f'A {formatted_name} - a large plant that provides oxygen, shade, and habitat for many creatures.'
    
    # Electronics
    elif 'computer' in label_lower or 'laptop' in label_lower:
        return f'A {formatted_name} - an electronic device used for computing, communication, and entertainment.'
    elif 'phone' in label_lower or 'smartphone' in label_lower:
        return f'A {formatted_name} - a portable electronic device used for communication and many other functions.'
    
    # Default
    else:
        return f'A {formatted_name} - an interesting object that has been identified in the image.'

def use_fallback_classification(image_bytes, reason='unknown'):
    """Fallback classification using improved image analysis"""
    try:
        # Open image with PIL
        img = Image.open(io.BytesIO(image_bytes))
        
        # Get image properties
        width, height = img.size
        aspect_ratio = width / height if height > 0 else 1
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize for faster processing if image is very large
        if width * height > 1000000:  # If more than 1M pixels
            img.thumbnail((500, 500), Image.Resampling.LANCZOS)
            width, height = img.size
        
        # Get pixel data
        pixels = list(img.getdata())
        total_pixels = len(pixels)
        
        # Calculate average color
        avg_r = sum(p[0] for p in pixels) / total_pixels
        avg_g = sum(p[1] for p in pixels) / total_pixels
        avg_b = sum(p[2] for p in pixels) / total_pixels
        
        # Calculate color variance (helps detect if image has distinct objects)
        var_r = sum((p[0] - avg_r) ** 2 for p in pixels) / total_pixels
        var_g = sum((p[1] - avg_g) ** 2 for p in pixels) / total_pixels
        var_b = sum((p[2] - avg_b) ** 2 for p in pixels) / total_pixels
        color_variance = (var_r + var_g + var_b) / 3
        
        # Count orange/ginger/brown pixels (common cat colors)
        orange_pixels = sum(1 for p in pixels if 
                           (p[0] > 150 and p[0] > p[1] * 1.2 and p[0] > p[2] * 1.2) or  # Orange
                           (p[0] > 100 and p[1] > 80 and p[2] < 100 and p[0] > p[2] * 1.5) or  # Ginger
                           (p[0] > 80 and p[1] > 60 and p[2] < 80 and abs(p[0] - p[1]) < 40))  # Brown
        
        orange_ratio = orange_pixels / total_pixels if total_pixels > 0 else 0
        
        # Count gray pixels (common for cats, especially in shadows)
        gray_pixels = sum(1 for p in pixels if abs(p[0] - p[1]) < 20 and abs(p[1] - p[2]) < 20 and 
                         (p[0] + p[1] + p[2]) / 3 < 200)
        gray_ratio = gray_pixels / total_pixels if total_pixels > 0 else 0
        
        # Check for organic shapes (not uniform - suggests living things)
        # High variance suggests complex image with objects
        is_complex = color_variance > 2000
        
        # Classification logic
        confidence = 0.60
        
        # Calculate brightness (helps distinguish animals)
        brightness = (avg_r + avg_g + avg_b) / 3
        
        # Check for golden/yellow tones (common in golden retrievers)
        golden_pixels = sum(1 for p in pixels if 
                           p[0] > 150 and p[1] > 120 and p[2] < 100 and 
                           p[0] > p[1] * 0.9 and p[1] > p[2] * 1.5)
        golden_ratio = golden_pixels / total_pixels if total_pixels > 0 else 0
        
        # More conservative approach: distinguish between cats and dogs better
        # Dogs (especially golden retrievers) tend to have more uniform golden/yellow tones
        # Cats often have more varied patterns (tabby, spots, etc.)
        
        # Check for animals FIRST before other classifications
        # Lower thresholds to catch animals even with green backgrounds
        # IMPORTANT: Check animals BEFORE red/color-based classification
        
        # Strong animal indicators (high confidence)
        # Check for orange/ginger FIRST (common in cats) before other colors
        if orange_ratio > 0.10 and is_complex and color_variance > 2000:
            result = {
                'name': 'Cat',
                'category': 'Animal',
                'confidence': min(0.85, 0.70 + orange_ratio * 2.0),
                'description': 'A cat - a beloved domestic pet known for its independent nature, agility, and affectionate behavior. Cats come in many colors including orange, ginger, tabby, and many other beautiful patterns.'
            }
        elif golden_ratio > 0.15 and is_complex and brightness > 100:
            result = {
                'name': 'Dog',
                'category': 'Animal',
                'confidence': min(0.85, 0.70 + golden_ratio * 1.5),
                'description': 'A dog - a loyal companion and one of humanity\'s oldest friends, known for intelligence and devotion. Dogs come in many breeds and colors, including golden retrievers, labradors, and many others.'
            }
        # High orange/ginger with high variance (more pattern variation) suggests cat
        elif orange_ratio > 0.15 and color_variance > 2500:
            result = {
                'name': 'Cat',
                'category': 'Animal',
                'confidence': min(0.85, 0.70 + orange_ratio * 1.5),
                'description': 'A cat - a beloved domestic pet known for its independent nature, agility, and affectionate behavior. Cats come in many colors including orange, ginger, tabby, and many other beautiful patterns.'
            }
        # Moderate animal indicators - lower thresholds for green background scenarios
        elif (orange_ratio > 0.08 or golden_ratio > 0.10 or gray_ratio > 0.12) and is_complex:
            # If brightness is high and colors are more uniform, likely dog
            if brightness > 110 and (golden_ratio > 0.08 or (orange_ratio > 0.05 and color_variance < 5000)):
                result = {
                    'name': 'Dog',
                    'category': 'Animal',
                    'confidence': min(0.80, 0.70 + (orange_ratio + golden_ratio) * 1.5),
                    'description': 'A dog - a loyal companion and one of humanity\'s oldest friends, known for intelligence and devotion.'
                }
            # Otherwise, be conservative and say "Animal"
            else:
                result = {
                    'name': 'Animal',
                    'category': 'Animal',
                    'confidence': 0.75,
                    'description': 'An animal - a warm-blooded creature that has been identified in the image. This could be a cat, dog, or another pet.'
                }
        # Expanded classification for many more categories
        
        # Count various color ranges for better detection
        red_pixels = sum(1 for p in pixels if p[0] > 150 and p[0] > p[1] * 1.3 and p[0] > p[2] * 1.3)
        red_ratio = red_pixels / total_pixels if total_pixels > 0 else 0
        
        white_pixels = sum(1 for p in pixels if p[0] > 200 and p[1] > 200 and p[2] > 200)
        white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
        
        black_pixels = sum(1 for p in pixels if (p[0] + p[1] + p[2]) / 3 < 50)
        black_ratio = black_pixels / total_pixels if total_pixels > 0 else 0
        
        # IMPORTANT: Check for animals FIRST, even with green backgrounds
        # Animals often appear on grass/outdoor scenes with green backgrounds
        # Use VERY low thresholds here since green background can mask animal colors
        # If image is complex (has objects), ANY warm colors = likely animal
        has_animal_colors = (orange_ratio > 0.03 or golden_ratio > 0.03 or gray_ratio > 0.05) and is_complex
        
        # Green suggests plants/nature, BUT check for animals first with VERY low thresholds
        if avg_g > avg_r * 1.2 and avg_g > avg_b * 1.2:
            # If we have ANY animal-like colors AND complexity, prioritize animal over green
            # Also check if image has high complexity (suggests objects, not just grass)
            if has_animal_colors or (is_complex and color_variance > 3000):
                # Determine if it's more likely a dog or cat - use very low thresholds
                if golden_ratio > 0.03 and brightness > 90:
                    result = {
                        'name': 'Dog',
                        'category': 'Animal',
                        'confidence': min(0.85, 0.70 + max(golden_ratio, orange_ratio) * 3.0),
                        'description': 'A dog - a loyal companion and one of humanity\'s oldest friends, known for intelligence and devotion. Dogs often appear in outdoor settings with green grass backgrounds.'
                    }
                elif orange_ratio > 0.05 and color_variance > 1500:
                    result = {
                        'name': 'Cat',
                        'category': 'Animal',
                        'confidence': min(0.85, 0.70 + orange_ratio * 3.0),
                        'description': 'A cat - a beloved domestic pet known for its independent nature, agility, and affectionate behavior.'
                    }
                else:
                    # High complexity on green background = likely animal (not just grass)
                    # Be even more aggressive - if variance is high, it's almost certainly an animal
                    if color_variance > 4000:
                        result = {
                            'name': 'Animal',
                            'category': 'Animal',
                            'confidence': 0.80,
                            'description': 'An animal - a warm-blooded creature that has been identified in the image. This appears to be an animal in an outdoor or natural setting.'
                        }
                    elif color_variance > 3000:
                        result = {
                            'name': 'Animal',
                            'category': 'Animal',
                            'confidence': 0.75,
                            'description': 'An animal - a warm-blooded creature that has been identified in the image. This appears to be an animal in an outdoor or natural setting.'
                        }
                    else:
                        # Medium complexity - still likely animal
                        result = {
                            'name': 'Animal',
                            'category': 'Animal',
                            'confidence': 0.72,
                            'description': 'An animal - a warm-blooded creature that has been identified in the image. This appears to be an animal in an outdoor or natural setting.'
                        }
            elif is_complex and color_variance < 2000:
                # Complex but low variance green scene - likely nature/plant
                result = {
                    'name': 'Plant or Nature Scene',
                    'category': 'Plant',
                    'confidence': 0.75,
                    'description': 'A plant or natural scene - living organisms that grow in soil and produce their own food through photosynthesis, or a beautiful natural landscape.'
                }
            else:
                # If it's complex but no clear animal colors, still prefer "Complex Object" over "Green Object"
                # Only use "Green Object" for truly simple images
                if is_complex:
                    result = {
                        'name': 'Complex Object or Animal',
                        'category': 'Object',
                        'confidence': 0.70,
                        'description': 'A complex object or scene that has been identified in the image. The image contains varied colors and patterns, possibly including an animal or other object on a green background.'
                    }
                else:
                    # Truly simple green object
                    result = {
                        'name': 'Green Object',
                        'category': 'Object',
                        'confidence': 0.70,
                        'description': 'A green-colored object that has been identified in the image.'
                    }
        # Blue suggests sky/water/technology
        elif avg_b > avg_r * 1.2 and avg_b > avg_g * 1.2:
            if brightness > 150:
                result = {
                    'name': 'Sky or Water',
                    'category': 'Nature',
                    'confidence': 0.75,
                    'description': 'A natural scene featuring sky or water - beautiful elements of our natural world.'
                }
            else:
                result = {
                    'name': 'Blue Object',
                    'category': 'Object',
                    'confidence': 0.70,
                    'description': 'A blue-colored object that has been identified in the image.'
                }
        # Red suggests food, flowers, or red objects
        # BUT check for animals first - cats can have orange/reddish tones
        elif red_ratio > 0.15 and is_complex:
            # If we have animal colors (orange/ginger) along with red, it's likely an animal
            if orange_ratio > 0.05 or golden_ratio > 0.05:
                # Likely a cat or dog with reddish/orange coloring
                if orange_ratio > golden_ratio:
                    result = {
                        'name': 'Cat',
                        'category': 'Animal',
                        'confidence': min(0.85, 0.70 + orange_ratio * 2.0),
                        'description': 'A cat - a beloved domestic pet known for its independent nature, agility, and affectionate behavior. Cats come in many colors including orange, ginger, and reddish tones.'
                    }
                else:
                    result = {
                        'name': 'Animal',
                        'category': 'Animal',
                        'confidence': 0.75,
                        'description': 'An animal - a warm-blooded creature that has been identified in the image. This appears to be an animal with reddish or orange coloring.'
                    }
            else:
                # Pure red without animal colors = food or red object
                result = {
                    'name': 'Red Object or Food',
                    'category': 'Food' if brightness > 100 else 'Object',
                    'confidence': 0.75,
                    'description': 'A red-colored object that has been identified. This could be food, a flower, or another red item.'
                }
        # White/light objects
        elif white_ratio > 0.30:
            if is_complex:
                result = {
                    'name': 'Light Colored Object',
                    'category': 'Object',
                    'confidence': 0.70,
                    'description': 'A light-colored or white object that has been identified in the image.'
                }
            else:
                result = {
                    'name': 'White Background or Object',
                    'category': 'Object',
                    'confidence': 0.65,
                    'description': 'A white or light-colored background or object.'
                }
        # Dark/black objects
        elif black_ratio > 0.30:
            result = {
                'name': 'Dark Object',
                'category': 'Object',
                'confidence': 0.70,
                'description': 'A dark or black-colored object that has been identified in the image.'
            }
        # High complexity with warm colors - likely animal or complex object
        elif is_complex and (avg_r > 100 or orange_ratio > 0.05):
            if color_variance > 5000:
                result = {
                    'name': 'Complex Object',
                    'category': 'Object',
                    'confidence': 0.75,
                    'description': 'A complex object with varied colors and patterns that has been identified in the image.'
                }
            else:
                result = {
                    'name': 'Animal',
                    'category': 'Animal',
                    'confidence': 0.70,
                    'description': 'An animal - a living creature that has been identified in the image. The specific type may vary, but it appears to be a warm-blooded animal.'
                }
        # Medium complexity - could be many things
        elif is_complex:
            result = {
                'name': 'Object',
                'category': 'Object',
                'confidence': 0.70,
                'description': 'An object with moderate complexity that has been identified in the image. The image contains varied colors and patterns.'
            }
        # Low complexity - simple object or background
        else:
            # Determine dominant color
            if avg_r > avg_g and avg_r > avg_b:
                color_name = 'Red'
            elif avg_g > avg_r and avg_g > avg_b:
                color_name = 'Green'
            elif avg_b > avg_r and avg_b > avg_g:
                color_name = 'Blue'
            elif brightness > 200:
                color_name = 'Light'
            elif brightness < 50:
                color_name = 'Dark'
            else:
                color_name = 'Colored'
            
            result = {
                'name': f'{color_name} Object',
                'category': 'Object',
                'confidence': 0.65,
                'description': f'A {color_name.lower()}-colored object that has been identified in the image.'
            }
        
        result['source'] = 'fallback'
        result['fallback_reason'] = reason

        return jsonify(result)
    except Exception as e:
        # Ultimate fallback
        return jsonify({
            'name': 'Unknown Object',
            'category': 'Unknown',
            'confidence': 0.50,
            'description': f'Unable to fully analyze the image. Error: {str(e)}',
            'source': 'fallback',
            'fallback_reason': 'exception'
        })

if __name__ == '__main__':
    print('🚀 PicDetect API Proxy Server')
    print('📡 Running on http://localhost:8001')
    print('🔗 This server proxies requests to Hugging Face API')
    print('⏹️  Press Ctrl+C to stop\n')
    app.run(host='0.0.0.0', port=8001, debug=False)

