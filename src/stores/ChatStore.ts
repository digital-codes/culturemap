// example from https://github.com/brillout/vite-plugin-ssr_legacy/blob/main/examples/vue-pinia/stores/useTodos.js

import { defineStore } from 'pinia'

interface Message {
  id: number | null
  type: string  // 'bot' | 'user'
  text: string
}

export const useChatStore = defineStore('chat', {
  state: () => ({
    messages: [] as Message[],
    cardId: -1,
    sessionId: '', // Optional: to track user sessions if needed
    convId: '' // Optional: to track conversation threads if needed
  }),
  getters: {
    messageById: (state) => (id: number) => state.messages.find((message) => message.id === id),
    allMessages: (state) => state.messages,
    getId: (state) => state.cardId,
    getSessionId: (state) => state.sessionId,
    getConvId: (state) => state.convId
  },
  actions: {
    clear() {
      this.messages = [] as Message[]
      this.cardId = -1
      this.sessionId = ''
      this.convId = ''
    },
    append(message: Message) {
      const newId = this.messages.length > 0 ? (this.messages[this.messages.length - 1]?.id ?? -1) + 1 : 0
      message.id = newId
      this.messages.push(message)
    },
    setId(id: number) {
      this.cardId = id
    },
    setSessionId(sessionId: string) {
      this.sessionId = sessionId
    },
    setConvId(convId: string) {
      this.convId = convId
    },
    // testing only
    async fetchMessages() {
      // Simulate an API call with a delay
      await new Promise((resolve) => setTimeout(resolve, 1000))
      const testMsgs: Message[] = [
        { id: 0, type: 'user', text: 'Buy milk' },
        { id: 1, type: 'bot', text: 'Give money' }
      ]
      testMsgs.forEach((msg) => this.append(msg)) // Use the append action to add messages to the state
    }
  }
})
