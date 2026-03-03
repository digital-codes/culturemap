<script setup lang="js">
import { ref, watch, nextTick } from 'vue';

import { useI18n } from 'vue-i18n';
const i18n = useI18n();

import { storeToRefs } from 'pinia'
import { useChatStore } from '../stores/ChatStore'

const chatStore = useChatStore()
const { messages } = storeToRefs(chatStore)

console.log('Initial messages in Chat.vue:', messages.value)

const query = ref('');

const sessionId = ref(chatStore.getSessionId);
const conversationId = ref(chatStore.getConvId);

watch(i18n.locale, (newLocale) => {
  console.log('Language changed to:', newLocale);
  // Optionally, you can also reset the chat or send a system message about the language change
  messages.value = [];
  sessionId.value = null;
  conversationId.value = null;
});


const submit = async () => {
  // Placeholder for submit logic
  console.log('Submit button clicked', query.value);
  try {
    const response = await fetch('/php/chat.php', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query.value,
        lang: i18n.locale.value,
        session: sessionId.value != '' ? sessionId.value : undefined,
        conversation_id: conversationId.value != '' ? conversationId.value : undefined
      })
    });
    const data = await response.json();
    console.log('Response from server:', data);
    if (!response.ok) {
      console.error('Server error:', data.error || response.statusText);
      throw new Error(data.error || 'Server error');
    }
    if (data.error) {
      throw new Error("Error: " + data.error);
    }
    // Update messages and session/conversation IDs based on response
    if (data.session) {
      sessionId.value = data.session;
      chatStore.setSessionId(data.session);
    }
    if (data.conversation_id) {
      conversationId.value = data.conversation_id;
      chatStore.setConvId(data.conversation_id);
    }
    chatStore.append({ type: 'question', text: query.value });
    chatStore.append({ type: 'answer', text: data.text || 'No response from server.' });
  } catch (error) {
    console.error('Error submitting query:', error);
  }
  query.value = '';
  await nextTick();
  const chatElement = document.querySelector('.chat');
  if (chatElement) {
    chatElement.scrollTop = chatElement.scrollHeight;
  }
};


</script>

<template>
  <div class="chat">
    <p v-for="(message, index) in messages" :key="index" :class="message.type">{{ message.text }}</p>
  </div>
  <div class="input-area">
    <input type="text" :placeholder="$t('message.typeYourMessage')" v-model="query" />
    <button @click="submit" >{{ $t("message.send") }}</button>
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
  height: 40vh;
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

.input-area {
  display: flex;
  gap: .5rem;
  margin: 1rem auto;
  width:80%;
}
.input-area input {
  flex: 1;
  padding: .5rem;
  border: 1px solid var(--color-border);
  border-radius: 4px;
}
.input-area button {
  padding: .5rem 1rem;
  background-color: var(--color-primary);
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

</style>    
