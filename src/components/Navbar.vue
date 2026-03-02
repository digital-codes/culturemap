<template>
    <div class="navbar">
        <nav>
            <button @click="menuOpen = !menuOpen" class="menu-toggle">Menu</button>
            <ul v-if="menuOpen" class="nav-menu">
                <li><a href="#" @click="menuActions[0]!.action(); menuOpen = false">Home</a></li>
                <li><a href="#" @click="menuActions[1]!.action(); menuOpen = false">Chat</a></li>
                <li><a href="#" @click="menuActions[2]!.action(); menuOpen = false">About</a></li>
            </ul>
        </nav>
        <div style="display: flex; align-items: center;width:70%;justify-content: center;">
        <span class="title right">Culture</span>
        <img :src="logo" alt="Logo" class="logo" />
        <span class="title">Map</span>
        </div>
        <button @click="tx = !tx" class="tx-toggle">
            {{ tx ? 'EN' : 'DE' }}
        </button>
        <button @click="darkMode = !darkMode" class="mode-toggle">
            {{ darkMode ? '☀️' : '🌙' }}
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

const emit = defineEmits(['toggleChat', 'toggleTx']);

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
    { name: 'Home', action: () => { console.log('Home clicked'); chatMode.value = false; } },
    { name: 'Chat', action: () => { console.log('Chat clicked'); chatMode.value = true; } },
    { name: 'About', action: () => { console.log('About clicked'); chatMode.value = false; } },
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

.title {
    font-size: 1.5rem;
    font-weight: bold;
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