// API Configuration
// Use local proxy server (runs on port 8001) to avoid CORS issues
const PROXY_API_URL = 'http://localhost:8001/classify';

// Fallback: Try direct Hugging Face API if proxy is not available
const HUGGINGFACE_API_URL = 'https://api-inference.huggingface.co/models/google/vit-base-patch16-224';

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const removeBtn = document.getElementById('removeBtn');
const classifyBtn = document.getElementById('classifyBtn');
const classifyIcon = classifyBtn ? classifyBtn.querySelector('.classify-icon') : null;
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const tryAgainBtn = document.getElementById('tryAgainBtn');
const thumbsUpBtn = document.getElementById('thumbsUp');
const thumbsDownBtn = document.getElementById('thumbsDown');
const ratingStatus = document.getElementById('ratingStatus');

// Theme toggle elements
const themeToggle = document.getElementById('themeToggle');
const themeToggleIcon = document.getElementById('themeToggleIcon');
const themeToggleLabel = themeToggle ? themeToggle.querySelector('.theme-toggle__label') : null;

const THEME_STORAGE_KEY = 'picdetect-theme';
const THEME_OVERRIDE_KEY = 'picdetect-theme-user';
const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)');

// Result elements
const resultIcon = document.getElementById('resultIcon');
const resultName = document.getElementById('resultName');
const resultCategory = document.getElementById('resultCategory');
const confidenceScore = document.getElementById('confidenceScore');
const confidenceFill = document.getElementById('confidenceFill');
const resultDescription = document.getElementById('resultDescription');
const funFact = document.getElementById('funFact');

let currentFile = null;
let currentResult = null;
let currentRating = null;

function applyTheme(theme) {
    const normalizedTheme = theme === 'dark' ? 'dark' : 'light';
    const isDark = normalizedTheme === 'dark';

    document.documentElement.setAttribute('data-theme', normalizedTheme);
    document.documentElement.classList.toggle('theme-dark', isDark);
    document.documentElement.classList.toggle('theme-light', !isDark);

    document.body.classList.toggle('theme-dark', isDark);
    document.body.classList.toggle('theme-light', !isDark);

    if (themeToggle) {
        themeToggle.setAttribute('aria-label', isDark ? 'Switch to light mode' : 'Switch to dark mode');
    }

    if (themeToggleLabel) {
        themeToggleLabel.textContent = isDark ? 'Light mode' : 'Dark mode';
    }

    if (themeToggleIcon) {
        themeToggleIcon.textContent = isDark ? 'light_mode' : 'dark_mode';
    }
}

function initializeTheme() {
    let storedTheme = null;

    try {
        storedTheme = localStorage.getItem(THEME_STORAGE_KEY);
    } catch (storageError) {
        storedTheme = null;
    }

    if (!storedTheme) {
        storedTheme = prefersDarkScheme.matches ? 'dark' : 'light';
    }

    applyTheme(storedTheme);
}

initializeTheme();

if (themeToggle) {
    themeToggle.addEventListener('click', () => {
        const nextTheme = document.body.classList.contains('theme-dark') ? 'light' : 'dark';
        const systemTheme = prefersDarkScheme.matches ? 'dark' : 'light';

        try {
            localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
            if (nextTheme === systemTheme) {
                localStorage.removeItem(THEME_OVERRIDE_KEY);
            } else {
                localStorage.setItem(THEME_OVERRIDE_KEY, 'true');
            }
        } catch (storageError) {
            // Ignore storage errors (private browsing, etc.)
        }

        applyTheme(nextTheme);
    });
}

function handleSystemThemeChange(event) {
    let userOverrode = false;

    try {
        userOverrode = localStorage.getItem(THEME_OVERRIDE_KEY) === 'true';
    } catch (storageError) {
        userOverrode = false;
    }

    if (userOverrode) {
        return;
    }

    const nextTheme = event.matches ? 'dark' : 'light';

    try {
        localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
    } catch (storageError) {
        // Ignore storage errors
    }

    applyTheme(nextTheme);
}

if (prefersDarkScheme) {
    if (typeof prefersDarkScheme.addEventListener === 'function') {
        prefersDarkScheme.addEventListener('change', handleSystemThemeChange);
    } else if (typeof prefersDarkScheme.addListener === 'function') {
        prefersDarkScheme.addListener(handleSystemThemeChange);
    }
}

function setRating(value) {
    if (!currentResult) {
        if (ratingStatus) {
            ratingStatus.textContent = 'Classify an image to leave a rating.';
        }
        return;
    }

    currentRating = value;

    if (thumbsUpBtn) thumbsUpBtn.classList.toggle('active', value === 'up');
    if (thumbsDownBtn) thumbsDownBtn.classList.toggle('active', value === 'down');

    if (ratingStatus) {
        ratingStatus.textContent = value === 'up' ? 'Glad it helped! 🌟' : 'Thanks, we’ll keep improving.';
    }
}

if (thumbsUpBtn) {
    thumbsUpBtn.addEventListener('click', () => setRating('up'));
}

if (thumbsDownBtn) {
    thumbsDownBtn.addEventListener('click', () => setRating('down'));
}

// Fun facts database
const funFacts = {
    // Animals
    'dog': 'Dogs have an incredible sense of smell - they can detect some odors in parts per trillion!',
    'cat': 'Cats have a special reflective layer behind their retinas called the tapetum lucidum, which makes their eyes glow in the dark.',
    'bird': 'Some birds can see ultraviolet light, which helps them find food and navigate.',
    'horse': 'Horses can sleep both standing up and lying down, thanks to a special locking mechanism in their legs.',
    'cow': 'Cows form deep bonds with their best friends and experience stress when separated.',
    'pig': 'Pigs are actually very clean animals and are smarter than dogs!',
    'sheep': 'Sheep have excellent memories and can remember up to 50 different sheep and human faces for years.',
    'chicken': 'Chickens can remember over 100 different faces of people and animals.',
    'duck': 'Ducks have waterproof feathers thanks to a special oil they produce.',
    'rabbit': 'Rabbits can see nearly 360 degrees around them without turning their heads.',
    'elephant': 'Elephants are the only mammals that can\'t jump!',
    'lion': 'A lion\'s roar can be heard from up to 5 miles away.',
    'tiger': 'Tigers have striped skin, not just striped fur!',
    'bear': 'Bears can run up to 30 miles per hour - that\'s faster than Usain Bolt!',
    'wolf': 'Wolves can hear sounds up to 6 miles away in the forest.',
    'fox': 'Foxes have whiskers on their legs that help them navigate in the dark.',
    'deer': 'Deer can jump up to 10 feet high and 30 feet long.',
    'squirrel': 'Squirrels plant thousands of new trees each year by forgetting where they buried their acorns.',
    'mouse': 'Mice can sense sadness in other mice and will comfort them.',
    'rat': 'Rats laugh when they\'re tickled, but at a frequency too high for humans to hear.',
    'hamster': 'Hamsters can store food in their cheeks equal to half their body size.',
    'guinea pig': 'Guinea pigs are not actually pigs - they\'re rodents!',
    'rabbit': 'Rabbits can see nearly 360 degrees around them without turning their heads.',
    'turtle': 'Some turtles can live for over 100 years!',
    'snake': 'Snakes can open their mouths up to 150 degrees to swallow large prey.',
    'lizard': 'Some lizards can detach their tails to escape predators, and grow new ones!',
    'frog': 'Frogs can freeze solid in winter and thaw out in spring, completely unharmed.',
    'fish': 'Fish can remember things for up to 5 months.',
    'shark': 'Sharks have been around for over 400 million years - longer than trees!',
    'dolphin': 'Dolphins have names for each other and can recognize themselves in mirrors.',
    'whale': 'Blue whales have hearts the size of small cars.',
    'octopus': 'Octopuses have three hearts: two pump blood to the gills, while the third pumps it to the rest of the body.',
    'butterfly': 'Butterflies use taste sensors on their feet to determine which plants are suitable for laying eggs.',
    'bee': 'Bees can recognize human faces by piecing together patterns!',
    'spider': 'Some spiders can produce silk that\'s stronger than steel.',
    'ant': 'Ants can carry objects up to 50 times their own body weight.',
    'sloth': 'Sloths can hold their breath for up to 40 minutes by slowing their heart rate.',
    'penguin': 'Penguins can drink salt water because they have a special gland that filters out the salt.',
    'polar bear': 'Polar bears have black skin under their white fur, which helps them absorb heat from the sun.',
    
    // Objects
    'car': 'The first car ever made could only go 10 miles per hour!',
    'bicycle': 'There are more bicycles in the world than cars - over 1 billion!',
    'airplane': 'A Boeing 747\'s wingspan is longer than the Wright brothers\' first flight distance.',
    'train': 'The longest train ever recorded was over 4.5 miles long!',
    'boat': 'The largest ship in the world can carry over 20,000 shipping containers.',
    'phone': 'The first mobile phone call was made in 1973 and the phone weighed 2.5 pounds!',
    'computer': 'The first computer weighed over 30 tons and took up an entire room.',
    'book': 'The world\'s smallest book is smaller than a grain of rice and can only be read with a microscope.',
    'chair': 'The average person sits for about 9.3 hours per day.',
    'table': 'The oldest table in the world is over 3,000 years old and was found in Egypt.',
    'lamp': 'The first light bulb could only last for about 13.5 hours.',
    'clock': 'The first mechanical clock was invented over 700 years ago.',
    'cup': 'The world\'s largest cup can hold over 2,000 gallons of liquid.',
    'bottle': 'It takes about 450 years for a plastic bottle to decompose.',
    'flower': 'Some flowers can change color based on the pH of the soil.',
    'tree': 'A single tree can absorb up to 48 pounds of carbon dioxide per year.',
    'house': 'The oldest house still standing is over 9,500 years old.',
    'building': 'The tallest building in the world is over 2,700 feet tall.',
    'bridge': 'The longest bridge in the world spans over 100 miles.',
    'mountain': 'Mountains are still growing - Mount Everest grows about 4mm per year.',
    'clock': 'The earliest mechanical clocks, developed in the 14th century, had no minute hand because telling time to the hour was precise enough.'
};

const descriptionHints = {
    'tiger': 'a powerful big cat with an orange coat and bold black stripes that help it blend into the jungle.',
    'lion': 'the “king of the jungle”, often spotted with a golden mane and a confident stride.',
    'dog': 'a loyal companion, usually eager-eyed and expressive. Notice the fur texture and posture.',
    'cat': 'a graceful feline, with curious eyes and agile movements. Watch for whiskers and soft fur patterns.',
    'elephant': 'a gentle giant, easily recognised by its long trunk and impressive tusks.',
    'horse': 'a strong, elegant animal with a muscular frame and flowing mane.',
    'bird': 'a feathered friend—look for wing shape and beak style to refine the species.',
    'flower': 'a vibrant bloom. Petal shapes and colors can help identify exact species.',
    'tree': 'a sturdy tree with branches and leaves forming its crown. Bark texture can refine the match.',
    'car': 'a sleek vehicle. Headlights, grille, and silhouette hint at the make and model.',
    'pizza': 'a delicious pizza with toppings and cheese textures that stand out.',
    'cake': 'a layered dessert—icing style and decorations can reveal the occasion.',
    'coffee': 'a comforting cup, with crema or foam giving extra clues.',
    'laptop': 'a portable computer. Keyboard layout and ports can pinpoint the brand.',
    'phone': 'a modern smartphone—note the camera placement and bezel size.',
    'clock': 'a clock face keeping track of each passing minute. Hands, numerals, and casing style can hint at the era or brand.'
};

// Emoji mapping for categories
const categoryIcons = {
    'animal': 'pets',
    'mammal': 'pets',
    'bird': 'emoji_nature',
    'reptile': 'emoji_nature',
    'amphibian': 'emoji_nature',
    'fish': 'emoji_nature',
    'insect': 'emoji_nature',
    'vehicle': 'directions_car',
    'food': 'restaurant',
    'plant': 'local_florist',
    'object': 'category',
    'furniture': 'weekend',
    'electronic': 'devices',
    'clothing': 'checkroom',
    'tool': 'build',
    'sport': 'sports_soccer',
    'musical instrument': 'music_note',
    'default': 'category'
};

// Event Listeners
uploadArea.addEventListener('click', () => fileInput.click());
uploadBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', handleFileSelect);

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFile(file);
    }
});

removeBtn.addEventListener('click', resetUpload);
classifyBtn.addEventListener('click', classifyImage);
tryAgainBtn.addEventListener('click', resetAll);

// Check if running from local file and show warning
(function() {
    if (window.location.protocol === 'file:') {
        const warning = document.getElementById('localFileWarning');
        if (warning) {
            warning.style.display = 'block';
        }
    }
})();

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewSection.style.display = 'block';
        resultsSection.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    currentFile = null;
    fileInput.value = '';
    previewSection.style.display = 'none';
    previewImage.src = '';
}

function resetAll() {
    resetUpload();
    resultsSection.style.display = 'none';
    loadingSection.style.display = 'none';
    currentResult = null;
    currentRating = null;
    if (ratingStatus) {
        ratingStatus.textContent = '';
    }
    if (thumbsUpBtn) thumbsUpBtn.classList.remove('active');
    if (thumbsDownBtn) thumbsDownBtn.classList.remove('active');
}

async function classifyImage() {
    if (!currentFile) return;

    // Show loading, hide preview and results
    loadingSection.style.display = 'block';
    previewSection.style.display = 'none';
    resultsSection.style.display = 'none';

    try {
        const base64Image = await fileToBase64(currentFile);
        const base64Only = base64Image.includes(',') ? base64Image.split(',')[1] : base64Image;

        // Try local proxy server first (most reliable)
        try {
            console.log('Trying local proxy server...');
            const response = await fetch(PROXY_API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: base64Only,
                    image_base64: base64Only
                })
            });

            if (response.ok) {
                const data = await response.json();
                console.log('Success with proxy server!');
                displayResults(data);
                return;
            } else {
                const errorData = await response.json().catch(() => ({}));
                if (response.status === 503 && errorData.retry_after) {
                    // Model loading, wait and retry
                    console.log(`Model loading, waiting ${errorData.retry_after} seconds...`);
                    await new Promise(resolve => setTimeout(resolve, errorData.retry_after * 1000));
                    const retryResponse = await fetch(PROXY_API_URL, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image: base64Only, image_base64: base64Only })
                    });
                    if (retryResponse.ok) {
                        const data = await retryResponse.json();
                        displayResults(data);
                        return;
                    }
                }
                throw new Error(`Proxy server error: ${response.status}`);
            }
        } catch (proxyError) {
            console.warn('Proxy server failed, trying direct API...', proxyError);
            
            // Fallback: Try direct Hugging Face API (may have CORS issues)
            try {
                const response = await fetch(HUGGINGFACE_API_URL, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        inputs: base64Only
                    })
                });

                if (response.ok) {
                    const hfData = await response.json();
                    if (Array.isArray(hfData) && hfData.length > 0) {
                        const topResult = hfData[0];
                        const label = topResult.label || 'Unknown';
                        const convertedData = {
                            name: formatLabel(label),
                            category: categorizeLabel(label),
                            confidence: topResult.score || 0.85,
                            description: generateDescription(label)
                        };
                        displayResults(convertedData);
                        return;
                    }
                }
            } catch (directError) {
                console.error('Direct API also failed:', directError);
            }
        }

        // If we get here, everything failed
        throw new Error('All classification methods failed');

    } catch (error) {
        console.error('Classification error:', error);
        showError('Unable to classify image. Please make sure the API proxy server is running (python3 api_proxy.py) and try again.');
    }
}

function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            // Remove data:image/...;base64, prefix if present
            const base64 = reader.result.split(',')[1] || reader.result;
            resolve(base64);
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

function generateMockClassification(file) {
    // This is a fallback function that generates mock data
    // In a real scenario, you'd remove this and use the actual API response
    const mockObjects = [
        { name: 'Golden Retriever', category: 'Mammal', confidence: 0.95, description: 'A friendly and intelligent dog breed known for its golden coat and gentle temperament.' },
        { name: 'Tabby Cat', category: 'Mammal', confidence: 0.92, description: 'A domestic cat with distinctive striped, dotted, or swirled patterns on its fur.' },
        { name: 'Red Rose', category: 'Plant', confidence: 0.88, description: 'A beautiful flowering plant symbolizing love and passion, known for its fragrant petals.' },
        { name: 'Mountain Bike', category: 'Vehicle', confidence: 0.90, description: 'A rugged bicycle designed for off-road cycling with enhanced durability and suspension.' },
        { name: 'Coffee Cup', category: 'Object', confidence: 0.85, description: 'A vessel designed for holding hot beverages, typically made of ceramic or porcelain.' }
    ];
    
    const random = mockObjects[Math.floor(Math.random() * mockObjects.length)];
    return {
        name: random.name,
        category: random.category,
        confidence: random.confidence,
        description: random.description
    };
}

function displayResults(data) {
    // Extract data from API response (adjust based on actual API structure)
    // Handle different possible response formats
    let name, category, confidence, description;
    
    if (Array.isArray(data) && data.length > 0) {
        // If response is an array, take the first (most confident) result
        const firstResult = data[0];
        name = firstResult.label || firstResult.name || firstResult.class || 'Unknown Object';
        confidence = firstResult.score || firstResult.confidence || 0.85;
        category = firstResult.category || firstResult.type || 'Object';
        description = firstResult.description || `A ${category.toLowerCase()} that has been identified in the image.`;
    } else if (data.results && Array.isArray(data.results)) {
        // If response has a results array
        const firstResult = data.results[0];
        name = firstResult.label || firstResult.name || firstResult.class || 'Unknown Object';
        confidence = firstResult.score || firstResult.confidence || 0.85;
        category = firstResult.category || firstResult.type || 'Object';
        description = firstResult.description || `A ${category.toLowerCase()} that has been identified in the image.`;
    } else {
        // Single object response
        name = data.name || data.label || data.class || data.object || 'Unknown Object';
        category = data.category || data.type || data.classification || 'Object';
        confidence = data.confidence || data.score || data.probability || 0.85;
        description = data.description || data.explanation || data.summary || `A ${category.toLowerCase()} that has been identified in the image.`;
    }

    // Set results
    resultName.textContent = name;
    resultCategory.textContent = category;
    confidenceScore.textContent = Math.round(confidence * 100);
    confidenceFill.style.width = `${confidence * 100}%`;
    const needsFriendlierCopy = !description || /interesting object/i.test(description);
    if (needsFriendlierCopy) {
        description = buildFriendlyDescription(name, category);
    }

    resultDescription.textContent = description;

    // Set icon based on category/name
    const categoryLower = category.toLowerCase();
    let iconName = categoryIcons.default;

    for (const [key, value] of Object.entries(categoryIcons)) {
        if (categoryLower.includes(key)) {
            iconName = value;
            break;
        }
    }

    const nameLower = name.toLowerCase();
    const nameIconMap = [
        { match: ['dog', 'puppy'], icon: 'pets' },
        { match: ['cat', 'kitten'], icon: 'pets' },
        { match: ['tiger', 'lion', 'leopard'], icon: 'cruelty_free' },
        { match: ['bird', 'eagle', 'parrot'], icon: 'emoji_nature' },
        { match: ['fish', 'dolphin', 'whale'], icon: 'water' },
        { match: ['car', 'truck', 'van'], icon: 'directions_car' },
        { match: ['bike', 'bicycle'], icon: 'directions_bike' },
        { match: ['airplane', 'plane', 'jet'], icon: 'flight' },
        { match: ['train'], icon: 'train' },
        { match: ['flower', 'rose'], icon: 'local_florist' },
        { match: ['tree'], icon: 'park' },
        { match: ['coffee', 'tea', 'cup'], icon: 'emoji_food_beverage' },
        { match: ['pizza', 'burger', 'sandwich'], icon: 'fastfood' },
        { match: ['cake', 'dessert'], icon: 'cake' },
        { match: ['laptop', 'computer'], icon: 'computer' },
        { match: ['phone', 'smartphone'], icon: 'smartphone' },
        { match: ['camera'], icon: 'photo_camera' },
        { match: ['guitar', 'piano'], icon: 'music_note' }
    ];

    for (const entry of nameIconMap) {
        if (entry.match.some(token => nameLower.includes(token))) {
            iconName = entry.icon;
            break;
        }
    }

    if (resultIcon) {
        resultIcon.textContent = iconName;
    }

    // Get fun fact
    const funFactKey = findFunFactKey(name, category);
    funFact.textContent = funFacts[funFactKey] || 
        `Did you know? ${name} is a fascinating ${category.toLowerCase()} with many interesting characteristics!`;

    // Show results, hide loading
    loadingSection.style.display = 'none';
    resultsSection.style.display = 'block';
    previewSection.style.display = 'block';

    currentResult = {
        name,
        category,
        confidence,
        description
    };

    currentRating = null;
    if (ratingStatus) {
        ratingStatus.textContent = '';
    }
    if (thumbsUpBtn) thumbsUpBtn.classList.remove('active');
    if (thumbsDownBtn) thumbsDownBtn.classList.remove('active');
}

function buildFriendlyDescription(name, category) {
    const nameText = name || 'the subject';
    const lowerName = nameText.toLowerCase();
    const lowerCategory = (category || '').toLowerCase();

    for (const [key, hint] of Object.entries(descriptionHints)) {
        if (lowerName.includes(key)) {
            return `This looks like ${nameText}, ${hint}`;
        }
    }

    const lead = `This appears to be ${nameText}`;

    if (lowerCategory.includes('animal') || [
        'dog','cat','tiger','lion','bird','horse','elephant','bear','wolf','fox','fish','dolphin','whale','penguin'
    ].some(animal => lowerName.includes(animal))) {
        return `${lead}, a living creature with its own personality. Tap the result for fun facts or try another image to compare.`;
    }

    if ([ 'flower','rose','tree','plant','leaf'].some(word => lowerName.includes(word)) || lowerCategory.includes('plant')) {
        return `${lead}, bringing a touch of nature. Notice the colors and texture for more precise matches.`;
    }

    if ([ 'pizza','burger','cake','coffee','meal','food','fruit','sushi','salad'].some(word => lowerName.includes(word)) || lowerCategory.includes('food')) {
        return `${lead}, and it looks delicious. If the dish has a specific style, you can capture it closer for an even more accurate label.`;
    }

    if ([ 'car','vehicle','bus','train','airplane','boat','bike','camera','phone','laptop'].some(word => lowerName.includes(word)) || lowerCategory.match(/vehicle|electronic|tool|furniture/)) {
        return `${lead}, a ${category ? category.toLowerCase() : 'useful item'} with recognizable structure and details.`;
    }

    if (lowerCategory.includes('nature')) {
        return `${lead}, part of a scenic outdoor moment. Lighting and composition help the model understand landscapes better.`;
    }

    return `${lead}. If you expected something else, try another angle or lighting so the model can compare more features.`;
}

function formatLabel(label) {
    // Convert labels like "tabby, tabby cat" or "Egyptian_cat" to readable format
    return label
        .replace(/_/g, ' ')
        .split(',')[0]
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
        .join(' ');
}

function categorizeLabel(label) {
    const labelLower = label.toLowerCase();
    
    // Animals
    if (labelLower.includes('cat') || labelLower.includes('dog') || labelLower.includes('bird') || 
        labelLower.includes('horse') || labelLower.includes('cow') || labelLower.includes('pig') ||
        labelLower.includes('sheep') || labelLower.includes('chicken') || labelLower.includes('duck') ||
        labelLower.includes('rabbit') || labelLower.includes('elephant') || labelLower.includes('lion') ||
        labelLower.includes('tiger') || labelLower.includes('bear') || labelLower.includes('wolf') ||
        labelLower.includes('fox') || labelLower.includes('deer') || labelLower.includes('squirrel') ||
        labelLower.includes('mouse') || labelLower.includes('rat') || labelLower.includes('hamster') ||
        labelLower.includes('turtle') || labelLower.includes('snake') || labelLower.includes('lizard') ||
        labelLower.includes('frog') || labelLower.includes('fish') || labelLower.includes('shark') ||
        labelLower.includes('dolphin') || labelLower.includes('whale') || labelLower.includes('octopus') ||
        labelLower.includes('butterfly') || labelLower.includes('bee') || labelLower.includes('spider') ||
        labelLower.includes('ant') || labelLower.includes('sloth') || labelLower.includes('penguin') ||
        labelLower.includes('polar bear')) {
        return 'Animal';
    }
    
    // Vehicles
    if (labelLower.includes('car') || labelLower.includes('truck') || labelLower.includes('bus') ||
        labelLower.includes('bicycle') || labelLower.includes('bike') || labelLower.includes('motorcycle') ||
        labelLower.includes('airplane') || labelLower.includes('train') || labelLower.includes('boat') ||
        labelLower.includes('ship')) {
        return 'Vehicle';
    }
    
    // Plants
    if (labelLower.includes('flower') || labelLower.includes('rose') || labelLower.includes('tree') ||
        labelLower.includes('plant') || labelLower.includes('leaf') || labelLower.includes('grass')) {
        return 'Plant';
    }
    
    // Food
    if (labelLower.includes('apple') || labelLower.includes('banana') || labelLower.includes('bread') ||
        labelLower.includes('pizza') || labelLower.includes('burger') || labelLower.includes('food')) {
        return 'Food';
    }
    
    // Electronics
    if (labelLower.includes('computer') || labelLower.includes('laptop') || labelLower.includes('phone') ||
        labelLower.includes('television') || labelLower.includes('tv') || labelLower.includes('camera')) {
        return 'Electronic';
    }
    
    return 'Object';
}

function generateDescription(label) {
    const labelLower = label.toLowerCase();
    const formattedName = formatLabel(label);
    
    if (labelLower.includes('cat')) {
        return `A ${formattedName} - a beloved domestic pet known for its independent nature, agility, and affectionate behavior.`;
    } else if (labelLower.includes('dog')) {
        return `A ${formattedName} - a loyal companion and one of humanity's oldest friends, known for intelligence and devotion.`;
    } else if (labelLower.includes('bird')) {
        return `A ${formattedName} - a feathered creature capable of flight, known for its beautiful songs and diverse species.`;
    } else if (labelLower.includes('flower') || labelLower.includes('rose')) {
        return `A ${formattedName} - a beautiful flowering plant that adds color and fragrance to gardens and bouquets.`;
    } else if (labelLower.includes('car') || labelLower.includes('vehicle')) {
        return `A ${formattedName} - a motorized vehicle designed for transportation on roads.`;
    } else {
        return `A ${formattedName} - an interesting object that has been identified in the image.`;
    }
}

function showError(message) {
    loadingSection.style.display = 'none';
    resultsSection.style.display = 'block';
    previewSection.style.display = 'block';
    
    resultName.textContent = 'Error';
    if (resultIcon) {
        resultIcon.textContent = 'error';
    }
    resultCategory.textContent = 'Error';
    confidenceScore.textContent = '0';
    confidenceFill.style.width = '0%';
    resultDescription.textContent = message;
    funFact.textContent = 'Please try uploading the image again or check your internet connection.';
}

function findFunFactKey(name, category) {
    const searchTerms = name.toLowerCase().split(' ');
    const categoryLower = category.toLowerCase();
    
    // Try exact matches first
    for (const term of searchTerms) {
        if (funFacts[term]) {
            return term;
        }
    }
    
    // Try partial matches
    for (const [key, value] of Object.entries(funFacts)) {
        if (name.toLowerCase().includes(key) || key.includes(searchTerms[0])) {
            return key;
        }
    }
    
    // Try category-based match
    if (categoryLower.includes('animal') || categoryLower.includes('mammal')) {
        return 'dog'; // Default animal fact
    }
    
    return null;
}

