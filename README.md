# IntroCD2-Novelty Detection

## Clone

```
cd existing_repo
git remote add origin https://gitlab.com/mab0205/introcd2-novelty-detection.git
git branch -M main
git push -uf origin main
```


## Description
This project aims to develop a novelty detection system applied to sports news. Novelty detection is a machine learning task that identifies new content by comparing it to previously known information.


### 🔜 Tasks
1. **Novelty Detection**  
   ⚠️ Identify `novel` and `non-novel` news articles by assessing the information in each target article relative to the source articles.

2. **System Development**  
   ⚠️ Develop a system that uses the `source` folder to distinguish between `novel` and `non-novel` information in the new articles within the `target` folder. This involves creating a model that can reliably recognize new information based on content changes detected through `novelty detection algorithms`.

3. **Evaluation and Comparison**  
   ⚠️ Compare the system's results with the `Document Level Annotation (DLA)` attribute in each target article to assess the model’s accuracy using the f1-score. This comparison will validate the system’s ability to label articles accurately as "novel" or "non-novel."

Each article has a `.txt` file with the content and an accompanying `.xml` file containing metadata such as title, publication date, publisher, and other event-related information.
## Authors and acknowledgment

nomes: Martín Ávila Buitrón
RAs: 2274183
logins GitLab:mab0205  

