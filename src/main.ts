import { createApp } from 'vue'
import './style.css'
import App from './App.vue'

import { createI18n } from 'vue-i18n'
import messages from './locales/messages.json'

import { createPinia } from 'pinia'
const pinia = createPinia()

const i18n = createI18n({
  legacy: false,
  locale: 'de',
  fallbackLocale: 'de',
  messages: messages,
})


const app = createApp(App)
app.use(i18n)
app.use(pinia)
app.mount('#app')
