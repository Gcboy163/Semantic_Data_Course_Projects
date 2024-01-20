# Chatting with Books using LLM.

The process involves using embeddings, which represent words or sentences as vectors, allowing computers to compare the meaning of different texts. This project uses the OpenAI embeddings API to obtain embeddings and PineCone, a cloud-based vector database, to store and search them.

### Basic Steps:

1. Load our data and split it up into small parts;
2. For each small part, send it to the OpenAI embeddings API and get the embedding back;
3. Store the embedding in a vector database, specifically designed to store and compare/search embeddings;
4. When a user asks a question, convert the question to an embedding and search the vector database for the closest embeddings, indicating that their meaning is related to the user’s question.

### Results:

#### BOOK: How-to-Succeed

**Q1**: What is the fastest way to get rich?

**A1**: The fastest way to get rich is to study yourself and your own wants, and find a way to supply a great want of humanity or improve existing methods. It is important to choose a vocation that is helpful to the largest number of people and to avoid selfish or detrimental occupations. Additionally, practicing strict economy and saving money can also lead to wealth.

**Q2**: What is the problem with most people?

**A2**: The problem with most people is that they are not living up to their full potential and are not fulfilling their individuality. They are not content with themselves and are constantly searching for external validation and possessions, leading to a superficial and unsatisfying life. They also lack moderation and self-awareness, causing them to becomd demoralized and unable to reach their goals.

**Q3**: What is the best way to peel bananas?

**A3**: I don't know.

#### BOOK: After-the-Cure

**Q1**: How to stop a plague?

**A1**: There are several ways to stop a plague, depending on the specific situation and type of plague. Some possible methods include implmenting quarantine measures, developing a cure or vaccine, improving sanitation and hygiene practices, and educating the public about prevention and treatment methods. It is also important to identify and address the root causes of the plague, such as poor living conditions or lack of access to healthcare. In some cases, international cooperation and aid may be necessary to effectively stop a plague.

**Q2**: How does the world look like after a plague?

**A2**: The world after a plague is described as broken and filled with Infected zones. People are struggling to survive and there is a lack of resources. The social fabric has also deteriorated, leading to violence and chaos. The remaining population is trying to pick up the pieces and move forward, but it is a difficult and uncertain process.

**Q3**: What are the symptoms of being infected?

**A3**: The symptoms of being infected include sore throat, colds, earaches, rashes, shambling or uncoordinated movement, aggression, and uncontrollable pica (cannibalism).

**Q4**: Are humans good creatures?

**A4**: I don't know.


