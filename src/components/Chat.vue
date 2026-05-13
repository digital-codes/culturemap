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
const isLoading = ref(false);
const error = ref('');

const sessionId = ref(chatStore.getSessionId);
const conversationId = ref(chatStore.getConvId);

watch(i18n.locale, (newLocale) => {
  console.log('Language changed to:', newLocale);
  messages.value = [];
  sessionId.value = null;
  conversationId.value = null;
  error.value = '';
});


const submit = async () => {
  const trimmedQuery = query.value.trim();
  
  if (!trimmedQuery) {
    error.value = i18n.t('message.enterMessage');
    return;
  }

  error.value = '';
  isLoading.value = true;

  try {
    const response = await fetch('/php/chat.php', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: trimmedQuery,
        lang: i18n.locale.value,
        session: sessionId.value !== '' && sessionId.value ? sessionId.value : undefined,
        conversation_id: conversationId.value !== '' && conversationId.value ? conversationId.value : undefined
      })
    });
    const data = await response.json();
    console.log('Response from server:', data);
    if (!response.ok) {
      throw new Error(data.error || i18n.t('message.serverError') || response.statusText);
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
    chatStore.append({ type: 'question', text: trimmedQuery });
    chatStore.append({ type: 'answer', text: data.text || i18n.t('message.noResponse') || 'No response from server.' });
  } catch (error) {
    console.error('Error submitting query:', error);
    error.value = error.message;
  } finally {
    isLoading.value = false;
    query.value = '';
    await nextTick();
    const chatElement = document.querySelector('.chat');
    if (chatElement) {
      chatElement.scrollTop = chatElement.scrollHeight;
    }
  }
};


</script>

<template>
  <!-- role="log" implies aria-live="polite" aria-atomic="false"; only newly appended content is announced -->
  <div class="chat" role="log">
    <p v-for="(message, index) in messages" :key="index" :class="message.type">{{ message.text }}</p>
  </div>
  <div class="input-area">
    <label for="chat-input" class="visually-hidden">{{ $t('message.typeYourMessage') }}</label>
    <input 
      id="chat-input" 
      type="text" 
      :placeholder="$t('message.typeYourMessage')" 
      v-model="query" 
      @keydown.enter.prevent="submit"
      :disabled="isLoading"
    />
    <button
      @click="submit"
      :aria-label="$t('message.send')"
      :disabled="isLoading || !query.trim()"
      :class="{ 'loading': isLoading }"
    >
      <span v-if="isLoading" class="spinner" aria-hidden="true"></span>
      <span v-else>{{ $t("message.send") }}</span>
    </button>
  </div>
  <div v-if="error" role="alert" class="error">{{ error }}</div>
</template>

<style scoped>
.chat {
  margin: 2rem auto;
  padding: 0;   
  background-color: var(--color-surface);
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0,  0, 0.1);
  overflow: scroll;
  height: 40vh;
}
.answer {
  color: var(--color-primary);
  margin-bottom: .2rem;
  margin-left: 2rem;
  width: 80%;
}
.question {
  color: var(--color-text);
  font-weight: bold;
  margin-right: .2rem;
  width: 80%;
}

.input-area {
  display: flex;
  gap: .5rem;
  margin: 1rem auto;
  width: 80%;
}
.input-area input {
  flex: 1;
  padding: .5rem;
  border: 1px solid var(--color-border);
  border-radius: 4px;
}
.input-area input:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
.input-area button {
  padding: .5rem 1rem;
  background-color: var(--color-primary);
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: .5rem;
  transition: background-color 0.2s, opacity 0.2s;
}
.input-area button:hover:not(:disabled) {
  background-color: var(--color-primary);
  opacity: 0.9;
}
.input-area button:focus-visible {
  outline: 2px solid var(--color-primary);
  outline-offset: 2px;
}
.input-area button:disabled {
  background-color: var(--color-border);
  cursor: not-allowed;
  opacity: 0.7;
}
.input-area button.loading {
  background-color: var(--color-border);
}

.spinner {
  width: 1rem;
  height: 1rem;
  border: 2px solid currentColor;
  border-right-color: transparent;
  border-radius: 50%;
  animation: spin 0.75s linear infinite;
}

@keyframes spin {
  100% { transform: rotate(360deg); }
}

.visually-hidden {
  position: absolute;
  width: 1px;
  height: 1px;
  margin: -1px;
  padding: 0;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

.error {
  color: red;
  font-size: 0.875rem;
  margin-top: 0.5rem;
}
</style>
