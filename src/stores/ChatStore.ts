// example from https://github.com/brillout/vite-plugin-ssr_legacy/blob/main/examples/vue-pinia/stores/useTodos.js

import { defineStore } from 'pinia'

interface Message {
  id: number
  text: string
}

export const useChatStore = defineStore('chat', {
  state: () => ({
    messages: [] as Message[]
  }),
  getters: {
    messageById: (state) => (id: number) => state.messages.find((message) => message.id === id)
  },
  actions: {
    async fetchMessages() {
      // simulate an API response
      const result: Message[] = await new Promise((resolve) => setTimeout(() => resolve(messages), 250))
      this.messages = result
    },
    async fetchMessageById(id: number) {
      const result: Message | undefined = await new Promise((resolve) => setTimeout(() => resolve(messages.find((message) => message.id === id))))
      if (result) {
        this.messages = [result]
      } else {
        this.messages = []
      }
    }
  }
})

const messages = [
  {
    id: 0,
    text: 'Buy milk'
  },
  {
    id: 1,
    text: 'Buy chocolate'
  }
]
