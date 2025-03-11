# Segmentarea tumorilor cerebrale prin imagini RMN

## Introducere
  Acest program își propune segmentarea unor imagini utilizând o arhitectură de învățare profundă și anume U-Net. Această arhitectură prezintă două „căi” : calea de contractare (blocul encoder) și calea expansivă (blocul decoder). Calea de contractare conține straturi de codificare care captează informații contextuale și reduc rezoluția spațială a intrării, în timp ce calea expansivă conține straturi de decodor care decodifică datele codificate și folosesc informațiile din calea de contractare prin conexiuni de ignorare pentru a genera o hartă de segmentare. 

## Setul de date
Setul de date conține imaginile RMN, cât și măștile corespunzătoare acestora. În faza de antrenare, se vor folosi 20% din imagini, iar restul de 80% se vor folosi pentru predicție. 

## Rulare
La rulare, asigurați-vă că programul conține dependențele care sunt menționate în requirements.txt.

