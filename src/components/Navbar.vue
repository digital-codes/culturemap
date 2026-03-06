<template>
    <div class="navbar">
        <nav class="nav" aria-label="Main navigation">
            <button
              @click="menuOpen = !menuOpen"
              class="menu-toggle"
              :aria-expanded="menuOpen"
              aria-controls="nav-menu"
              :aria-label="menuOpen ? $t('message.closeMenu') : $t('message.openMenu')"
            >
              <span class="icon" aria-hidden="true">{{ menuOpen ? 'close' : 'menu' }}</span>
            </button>
            <ul v-if="menuOpen" id="nav-menu" class="nav-menu">
                <li><button @click="menuActions[0]!.action(); menuOpen = false">{{ $t('message.home') }}</button></li>
                <li><button @click="menuActions[1]!.action(); menuOpen = false">{{ $t('message.chat') }}</button></li>
                <li><button @click="menuActions[2]!.action(); menuOpen = false">{{ $t('message.about') }}</button></li>
                <li><button @click="menuActions[3]!.action(); menuOpen = false">{{ $t('message.edit') }}</button></li>
            </ul>
        </nav>
        <div style="display: flex; align-items: center;width:70%;justify-content: center;" aria-hidden="true">
        <span class="title right">Culture</span>
        <img :src="logo" alt="Culture Map Karlsruhe logo" class="logo" />
        <span class="title">Map</span>
        </div>
        <button @click="tx = !tx" class="tx-toggle" :aria-label="tx ? $t('message.switchToDe') : $t('message.switchToEn')" :aria-pressed="tx">
            {{ tx ? 'EN' : 'DE' }}
        </button>
        <button @click="darkMode = !darkMode" class="mode-toggle" :aria-label="darkMode ? $t('message.switchToLight') : $t('message.switchToDark')" :aria-pressed="darkMode">
            <span class="icon" aria-hidden="true">{{ darkMode ? 'light_mode' : 'dark_mode' }}</span>
        </button>
    </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue';
import logo from '../assets/img/logo.jpg';
const darkMode = ref(false);
const menuOpen = ref(false);

const chatMode = ref(false);
const tx = ref(false);

const emit = defineEmits(['toggleChat', 'toggleTx',"route"]);

watch(chatMode, (newValue) => {
    emit('toggleChat', newValue);
});

watch(tx, (newValue) => {
    emit('toggleTx', newValue);
});


watch(darkMode, (newValue) => {
    if (newValue) {
        document.documentElement.classList.add('dark');
    } else {
        document.documentElement.classList.remove('dark');
    }
});

const menuActions = [
    { name: 'Home', action: () => { console.log('Home clicked'); emit("route", "home"); chatMode.value = false; } },
    { name: 'Chat', action: () => { console.log('Chat clicked'); emit("route", "chat"); chatMode.value = true; } },
    { name: 'About', action: () => { console.log('About clicked'); emit("route", "about"); chatMode.value = false; } },
    { name: 'Edit', action: () => { console.log('Edit clicked'); emit("route", "edit"); chatMode.value = false; } },
];


</script>

<style scoped>

.navbar {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 3rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem;
    background-color: var(--color-surface);
    color: var(--color-text);
    z-index:99;
}

.nav-menu {
    list-style: none;
    display: flex;
    gap: 1rem;
    position: fixed;
    top: 3rem;
    left: 1rem;
    background-color: var(--color-surface);
    width: 30%;
    padding: 1rem;
}

.nav-menu button {
    background: none;
    border: none;
    color: var(--color-text);
    cursor: pointer;
    font-size: 1rem;
    padding: 0;
    text-decoration: underline;
}

.title {
    font-family: "Faster One";
    font-size: 1.5rem;
    /* font-weight: bold; */
    width: 40%;
    overflow: clip;
    text-align: left;
}

.title.right {
    text-align: right;
}


.logo {
    height: 3em;
    padding: .25em;
    will-change: filter;
    transition: filter 300ms;
}

.logo:hover {
    filter: drop-shadow(0 0 2em #646cffaa);
}

.logo.vue:hover {
    filter: drop-shadow(0 0 2em #42b883aa);
}

.mode-toggle {
    margin-right: 2rem;
    cursor: pointer;
}

.tx-toggle {
    margin-right: 1rem;
    cursor: pointer;
}

.menu-toggle {
    margin-right: 2rem;
    margin-left: 2rem;
    cursor: pointer;
}

@media screen and (max-width: 600px) {
    .logo {
        height: 3em;
        padding: .15em;
    }

    .navbar {
        height: 2rem;
        padding: .5rem;
    }
    .nav-menu {
        width: 80%;
        top: 2rem;
        left: .5rem;
    }

    .title {
        font-size: 1.2rem;
        width:50%;
    }

.mode-toggle {
    margin-right: 1rem;
}

.tx-toggle {
    margin-right: .5rem;
}

.menu-toggle {
    margin-right: 1rem;
    margin-left: 1rem;
}

}

</style>