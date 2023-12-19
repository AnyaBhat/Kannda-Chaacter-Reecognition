from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from keras.models import load_model
import cv2
import numpy as np
import pytesseract
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load the trained model
model_path = 'D:\Final_year_project\Model\kannada_model.h5'
loaded_model = load_model(model_path)

# Class to character mapping
class_to_char_mapping = {
1:'ಅ',
2:'ಆ',
3:'ಇ',
4:'ಈ',
5:'ಉ',
6:'ಊ',
7:'ಋ',
8:'ೠ',
9:'ಎ',
10:'ಏ',
11:'ಐ',
12:'ಒ', 
13:'ಓ',
14:'ಔ',
15:'ಅಂ',
16:'ಅಃ',
17:'ಕ್',
18:'ಕ',
19:'ಕಾ',
20:'ಕಿ',
21:'ಕೀ',
22:'ಕು', 
23:'ಕೂ', 
24:'ಕೃ', 
25:'ಕೄ', 
26:'ಕೆ',
27:'ಕೇ', 
28:'ಕೈ', 
29:'ಕೊ', 
30:'ಕೋ', 
31:'ಕೌ', 
32:'ಕಂ', 
33:'ಕಃ',
34:'ಖ್',
35:'ಖ', 
36:'ಖಾ',
37:'ಖಿ', 
38:'ಖೀ', 
39:'ಖು', 
40:'ಖೂ',
41:'ಖೃ', 
42:'ಖೄ', 
43:'ಖೆ', 
44:'ಖೇ', 
45:'ಖೈ', 
46:'ಖೊ', 
47:'ಖೋ', 
48:'ಖೌ',
49:'ಖಂ', 
50:'ಖಃ',
51:'ಗ್',
52:'ಗ',
53:'ಗಾ',
54:'ಗಿ',
55:'ಗೀ',
56:'ಗು', 
57:'ಗೂ', 
58:'ಗೃ', 
59:'ಗೄ', 
60:'ಗೆ',
61:'ಗೇ', 
62:'ಗೈ', 
63:'ಗೊ', 
64:'ಗೋ', 
65:'ಗೌ', 
66:'ಗಂ', 
67:'ಗಃ',
68:'ಘ್',
69:'ಘ', 
70:'ಘಾ', 
71:'ಘಿ', 
72:'ಘೀ', 
73:'ಘು', 
74:'ಘೂ', 
75:'ಘೃ', 
76:'ಘೄ',
77:'ಘೆ', 
78:'ಘೇ', 
79:'ಘೈ', 
80:'ಘೊ', 
81:'ಘೋ', 
82:'ಘೌ', 
83:'ಘಂ', 
84:'ಘಃ',
85:'ಙ್',
86:'ಙ', 
87:'ಙಾ', 
88:'ಙಿ', 
89:'ಙೀ', 
90:'ಙು', 
91:'ಙೂ', 
92:'ಙೃ', 
93:'ಙೄ', 
94:'ಙೆ', 
95:'ಙೇ', 
96:'ಙೈ', 
97:'ಙೊ', 
98:'ಙೋ', 
99:'ಙೌ', 
100:'ಙಂ', 
101:'ಙಃ',
102:'ಚ್',
103:'ಚ', 
104:'ಚಾ', 
105:'ಚಿ', 
106:'ಚೀ', 
107:'ಚು', 
108:'ಚೂ', 
109:'ಚೃ', 
110:'ಚೄ', 
111:'ಚೆ', 
112:'ಚೇ', 
113:'ಚೈ', 
114:'ಚೊ', 
115:'ಚೋ', 
116:'ಚೌ', 
117:'ಚಂ', 
118:'ಚಃ',
119:'ಛ್',
120:'ಛ', 
121:'ಛಾ', 
122:'ಛಿ', 
123:'ಛೀ', 
124:'ಛು', 
125:'ಛೂ', 
126:'ಛೃ', 
127:'ಛೄ', 
128:'ಛೆ', 
129:'ಛೇ', 
130:'ಛೈ', 
131:'ಛೊ', 
132:'ಛೋ', 
133:'ಛೌ', 
134:'ಛಂ', 
135:'ಛಃ',
136:'ಜ್',
137:'ಜ', 
138:'ಜಾ',
139:'ಜಿ', 
140:'ಜೀ', 
141:'ಜು', 
142:'ಜೂ', 
143:'ಜೃ', 
144:'ಜೄ', 
145:'ಜೆ', 
146:'ಜೇ', 
147:'ಜೈ', 
148:'ಜೊ', 
149:'ಜೋ', 
150:'ಜೌ', 
151:'ಜಂ', 
152:'ಜಃ',
153:'ಝ್',
154:'ಝ', 
155:'ಝಾ', 
156:'ಝಿ', 
157:'ಝೀ', 
158:'ಝು', 
159:'ಝೂ', 
160:'ಝೃ', 
161:'ಝೄ', 
162:'ಝೆ', 
163:'ಝೇ', 
164:'ಝೈ',
165:'ಝೊ', 
166:'ಝೋ', 
167:'ಝೌ', 
168:'ಝಂ', 
169:'ಝಃ',
170:'ಞ್',
171:'ಞ', 
172:'ಞಾ', 
173:'ಞಿ', 
174:'ಞೀ', 
175:'ಞು', 
176:'ಞೂ',
177:'ಞೃ', 
178:'ಞೄ', 
179:'ಞೆ', 
180:'ಞೇ', 
181:'ಞೈ', 
182:'ಞೊ', 
183:'ಞೋ', 
184:'ಞೌ', 
185:'ಞಂ', 
186:'ಞಃ',
187:'ಟ್',
188:'ಟ', 
189:'ಟಾ', 
190:'ಟಿ', 
191:'ಟೀ', 
192:'ಟು', 
193:'ಟೂ', 
194:'ಟೃ', 
195:'ಟೄ', 
196:'ಟೆ', 
197:'ಟೇ', 
198:'ಟೈ', 
199:'ಟೊ', 
200:'ಟೋ', 
201:'ಟೌ', 
202:'ಟಂ', 
203:'ಟಃ',
204:'ಠ್',
205:'ಠ', 
206:'ಠಾ', 
207:'ಠಿ', 
208:'ಠೀ', 
209:'ಠು', 
210:'ಠೂ', 
211:'ಠೃ', 
212:'ಠೄ', 
213:'ಠೆ', 
214:'ಠೇ', 
215:'ಠೈ', 
216:'ಠೊ', 
217:'ಠೋ', 
218:'ಠೌ', 
219:'ಠಂ', 
220:'ಠಃ',
221:'ಡ್',
222:'ಡ', 
223:'ಡಾ', 
224:'ಡಿ', 
225:'ಡೀ', 
226:'ಡು', 
227:'ಡೂ', 
228:'ಡೃ', 
229:'ಡೄ', 
230:'ಡೆ', 
231:'ಡೇ', 
232:'ಡೈ', 
233:'ಡೊ', 
234:'ಡೋ', 
235:'ಡೌ', 
236:'ಡಂ', 
237:'ಡಃ',
238:'ಢ್',
239:'ಢ', 
240:'ಢಾ', 
241:'ಢಿ', 
242:'ಢೀ', 
243:'ಢು', 
244:'ಢೂ', 
245:'ಢೃ', 
246:'ಢೄ', 
247:'ಢೆ', 
248:'ಢೇ', 
249:'ಢೈ', 
250:'ಢೊ', 
251:'ಢೋ', 
252:'ಢೌ', 
253:'ಢಂ', 
254:'ಢಃ',
255:'ಣ್',
256:'ಣ', 
257:'ಣಾ', 
258:'ಣಿ', 
259:'ಣೀ', 
260:'ಣು', 
261:'ಣೂ', 
262:'ಣೃ', 
263:'ಣೄ', 
264:'ಣೆ', 
265:'ಣೇ', 
266:'ಣೈ', 
267:'ಣೊ', 
268:'ಣೋ', 
269:'ಣೌ', 
270:'ಣಂ', 
271:'ಣಃ',
272:'ತ್',
273:'ತ', 
274:'ತಾ', 
275:'ತಿ', 
276:'ತೀ', 
277:'ತು', 
278:'ತೂ', 
279:'ತೃ', 
280:'ತೄ', 
281:'ತೆ', 
282:'ತೇ', 
283:'ತೈ', 
284:'ತೊ', 
285:'ತೋ', 
286:'ತೌ', 
287:'ತಂ', 
288:'ತಃ',
289:'ಥ್',
290:'ಥ', 
291:'ಥಾ', 
292:'ಥಿ', 
293:'ಥೀ', 
294:'ಥು', 
295:'ಥೂ', 
296:'ಥೃ', 
297:'ಥೄ', 
298:'ಥೆ', 
299:'ಥೇ', 
300:'ಥೈ', 
301:'ಥೊ', 
302:'ಥೋ', 
303:'ಥೌ', 
304:'ಥಂ', 
305:'ಥಃ',
306:'ದ್',
307:'ದ', 
308:'ದಾ', 
309:'ದಿ', 
310:'ದೀ', 
311:'ದು', 
312:'ದೂ', 
313:'ದೃ', 
314:'ದೄ', 
315:'ದೆ', 
316:'ದೇ', 
317:'ದೈ', 
318:'ದೊ', 
319:'ದೋ', 
320:'ದೌ', 
321:'ದಂ', 
322:'ದಃ',
323:'ಧ್',
324:'ಧ', 
325:'ಧಾ', 
326:'ಧಿ', 
327:'ಧೀ', 
328:'ಧು', 
329:'ಧೂ', 
330:'ಧೃ', 
331:'ಧೄ', 
332:'ಧೆ', 
333:'ಧೇ', 
334:'ಧೈ', 
335:'ಧೊ', 
336:'ಧೋ', 
337:'ಧೌ', 
338:'ಧಂ', 
339:'ಧಃ',
340:'ನ್',
341:'ನ', 
342:'ನಾ', 
343:'ನಿ', 
344:'ನೀ', 
345:'ನು', 
346:'ನೂ', 
347:'ನೃ', 
348:'ನೄ', 
349:'ನೆ', 
350:'ನೇ', 
351:'ನೈ', 
352:'ನೊ', 
353:'ನೋ', 
354:'ನೌ', 
355:'ನಂ', 
356:'ನಃ',
357:'ಪ್',
358:'ಪ', 
359:'ಪಾ',
360:'ಪಿ', 
361:'ಪೀ', 
362:'ಪು', 
363:'ಪೂ',
364:'ಪೃ', 
365:'ಪೄ', 
366:'ಪೆ', 
367:'ಪೇ', 
368:'ಪೈ', 
369:'ಪೊ', 
370:'ಪೋ', 
371:'ಪೌ', 
372:'ಪಂ', 
373:'ಪಃ',
374:'ಫ್',
375:'ಫ', 
376:'ಫಾ', 
377:'ಫಿ', 
378:'ಫೀ', 
379:'ಫು', 
380:'ಫೂ',
381:'ಫೃ', 
382:'ಫೄ', 
383:'ಫೆ', 
384:'ಫೇ', 
385:'ಫೈ', 
386:'ಫೊ', 
387:'ಫೋ', 
388:'ಫೌ', 
389:'ಫಂ', 
390:'ಫಃ',
391:'ಬ್',
392:'ಬ', 
393:'ಬಾ', 
394:'ಬಿ', 
395:'ಬೀ',
396:'ಬು', 
397:'ಬೂ', 
398:'ಬೃ', 
399:'ಬೄ', 
400:'ಬೆ', 
401:'ಬೇ', 
402:'ಬೈ', 
403:'ಬೊ', 
404:'ಬೋ', 
405:'ಬೌ', 
406:'ಬಂ', 
407:'ಬಃ',
408:'ಭ್',
409:'ಭ', 
410:'ಭಾ', 
411:'ಭಿ', 
412:'ಭೀ', 
413:'ಭು', 
414:'ಭೂ', 
415:'ಭೃ', 
416:'ಭೄ', 
417:'ಭೆ', 
418:'ಭೇ', 
419:'ಭೈ', 
420:'ಭೊ', 
421:'ಭೋ', 
422:'ಭೌ', 
423:'ಭಂ', 
424:'ಭಃ',
425:'ಮ್',
426:'ಮ', 
427:'ಮಾ', 
428:'ಮಿ', 
429:'ಮೀ', 
430:'ಮು', 
431:'ಮೂ', 
432:'ಮೃ', 
433:'ಮೄ', 
434:'ಮೆ', 
435:'ಮೇ', 
436:'ಮೈ', 
437:'ಮೊ', 
438:'ಮೋ', 
439:'ಮೌ', 
440:'ಮಂ', 
441:'ಮಃ',
442:'ಯ್',
443:'ಯ', 
444:'ಯಾ', 
445:'ಯಿ', 
446:'ಯೀ', 
447:'ಯು', 
448:'ಯೂ', 
449:'ಯೃ', 
450:'ಯೄ', 
451:'ಯೆ', 
452:'ಯೇ', 
453:'ಯೈ', 
454:'ಯೊ', 
455:'ಯೋ', 
456:'ಯೌ', 
457:'ಯಂ', 
458:'ಯಃ',
459:'ರ್',
460:'ರ', 
461:'ರಾ', 
462:'ರಿ', 
463:'ರೀ', 
464:'ರು', 
465:'ರೂ', 
466:'ರೃ', 
467:'ರೄ', 
468:'ರೆ', 
469:'ರೇ', 
470:'ರೈ', 
471:'ರೊ', 
472:'ರೋ', 
473:'ರೌ', 
474:'ರಂ', 
475:'ರಃ',
476:'ಱ್',
477:'ಱ',
478:'ಱಾ',	
479:'ಱಿ',	
480:'ಱೀ',	
481:'ಱು',	
482:'ಱೂ',	
483:'ಱೃ',	
484:'ಱೄ',	
485:'ಱೆ',	
486:'ಱೇ',	
487:'ಱೈ',	
488:'ಱೊ',	
489:'ಱ',	
490:'ಱೌ',	
491:'ಱಂ',	
492:'ಱಃ',  	
493:'ಲ್',
494:'ಲ', 
495:'ಲಾ', 
496:'ಲಿ', 
497:'ಲೀ', 
498:'ಲು', 
499:'ಲೂ', 
500:'ಲೃ', 
501:'ಲೄ', 
502:'ಲೆ', 
503:'ಲೇ', 
504:'ಲೈ', 
505:'ಲೊ', 
506:'ಲೋ', 
507:'ಲೌ', 
508:'ಲಂ', 
509:'ಲಃ',
510:'ಳ್',
511:'ಳ', 
512:'ಳಾ', 
513:'ಳಿ', 
514:'ಳೀ', 
515:'ಳು', 
516:'ಳೂ', 
517:'ಳೃ', 
518:'ಳೄ', 
519:'ಳೆ', 
520:'ಳೇ', 
521:'ಳೈ', 
522:'ಳೊ', 
523:'ಳೋ', 
524:'ಳೌ', 
525:'ಳಂ', 
526:'ಳಃ',
527:'ೞ್',
528:'ೞ',
529:'ೞಾ',	
530:'ೞಿ',
531:'ೞೀ',	
532:'ೞು',	
533:'ೞೂ',	
534:'ೞೃ',	
535:'ೞೄ',	
536:'ೞೆ',	
537:'ೞೇ',	
538:'ೞೈ',	
539:'ೞೊ',	
540:'ೞೋ',	
541:'ೞೌ',	
542:'ೞಂ',	
543:'ೞಃ',	
544:'ವ್',
545:'ವ', 
546:'ವಾ', 
547:'ವಿ', 
548:'ವೀ', 
549:'ವು', 
550:'ವೂ', 
551:'ವೃ', 
552:'ವೄ', 
553:'ವೆ', 
554:'ವೇ', 
555:'ವೈ', 
556:'ವೊ', 
557:'ವೋ', 
558:'ವೌ', 
559:'ವಂ', 
560:'ವಃ',
561:'ಶ್',
562:'ಶ', 
563:'ಶಾ', 
564:'ಶಿ', 
565:'ಶೀ', 
566:'ಶು', 
567:'ಶೂ', 
568:'ಶೃ', 
569:'ಶೄ', 
570:'ಶೆ', 
571:'ಶೇ', 
572:'ಶೈ', 
573:'ಶೊ', 
574:'ಶೋ', 
575:'ಶೌ', 
576:'ಶಂ', 
577:'ಶಃ',
578:'ಷ್',
579:'ಷ', 
580:'ಷಾ', 
581:'ಷಿ', 
582:'ಷೀ', 
583:'ಷು', 
584:'ಷೂ', 
585:'ಷೃ', 
586:'ಷೄ', 
587:'ಷೆ', 
588:'ಷೇ', 
589:'ಷೈ', 
590:'ಷೊ', 
591:'ಷೋ', 
592:'ಷೌ', 
593:'ಷಂ', 
594:'ಷಃ',
595:'ಸ್',
596:'ಸ', 
597:'ಸಾ', 
598:'ಸಿ', 
599:'ಸೀ', 
600:'ಸು', 
601:'ಸೂ', 
602:'ಸೃ', 
603:'ಸೄ', 
604:'ಸೆ', 
605:'ಸೇ', 
606:'ಸೈ', 
607:'ಸೊ', 
608:'ಸೋ', 
609:'ಸೌ', 
610:'ಸಂ', 
611:'ಸಃ',
612:'ಹ್',
613:'ಹ', 
614:'ಹಾ', 
615:'ಹಿ', 
616:'ಹೀ', 
617:'ಹು', 
618:'ಹೂ', 
619:'ಹೃ', 
620:'ಹೄ', 
621:'ಹೆ', 
622:'ಹೇ', 
623:'ಹೈ', 
624:'ಹೊ', 
625:'ಹೋ', 
626:'ಹೌ', 
627:'ಹಂ', 
628:'ಹಃ',
629:'ಕ್ಷ್',
630:'ಕ್ಷ',
631:'ಕ್ಷಾ',
632:'ಕ್ಷಿ',
633:'ಕ್ಷೀ',
634:'ಕ್ಷು',
635:'ಕ್ಷೂ',
636:'ಕ್ಷೃ',
637:'ಕ್ಷೄ',
638:'ಕ್ಷೆ',
639:'ಕ್ಷೇ',
640:'ಕ್ಷೈ ',
641:'ಕ್ಷೊ',
642:'ಕ್ಷೋ',
643:'ಕ್ಷೌ',
644:'ಕ್ಷಂ',
645:'ಕ್ಷಃ',
646:'ೞ',
647:'ಱ',
648:'೦',
649:'೧',
650:'೨',
651:'೩',
652:'೪',
653:'೫',
654:'೬',
655:'೭',
656:'೮',
657:'೯',
    # ... Add entries for every character. Check in the data set and its CSV file. Each character has 25 images with a class.
}

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded image from the request
        image = request.files['image']
        
        # Read and preprocess the image
        img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
        img = cv2.resize(img, (64, 64))
        img = img / 255.0
        input_image_reshaped = img.reshape(-1, 64, 64, 1)

        # Make predictions for the image
        prediction = loaded_model.predict(input_image_reshaped)

        # Get the predicted class for the image
        predicted_class = np.argmax(prediction) + 1  # Add 1 to convert back to original class values

        # Map the predicted class to the Kannada character using the dictionary
        predicted_char = class_to_char_mapping.get(predicted_class, 'Unknown')

        # Return the predicted class and character as JSON
        return jsonify({
            'predicted_class': int(predicted_class),
            'predicted_char': predicted_char
        })

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict_labeled', methods=['POST'])
def predict_labeled():
    try:
        # Get the uploaded labeled image from the request
        labeled_image = request.files['image']

        # Use PyTesseract for OCR on labeled image
        labeled_image.seek(0)  # Reset file pointer
        labeled_text = pytesseract.image_to_string(Image.open(labeled_image), lang='kan')

        return jsonify({
            'predicted_text': labeled_text
        })

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
