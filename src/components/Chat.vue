<template>
  <div class="chat">
    <p class="answer">Hello! How can I assist you today?</p>
    <p class="question">What is this project about?</p>
  </div>
</template>

<style scoped>
.chat {
  margin: 2rem auto;
  padding: 0;   
  background-color: var(--color-surface);
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0,  0, 0.1);
  overflow:scroll;
  max-height: 200px;
}
.answer {
  color: var(--color-primary);
  margin-bottom: .2rem;
  margin-left: 2rem;
  width:80%;
}
.question {
  color: var(--color-text);
  font-weight: bold;
  margin-right: .2rem;
  width:80%;
}
</style>    
