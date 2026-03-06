<script setup lang="ts">
import { ref, onMounted, nextTick } from 'vue';

import Header from './components/Header.vue'
import Footer from './components/Footer.vue'
import Navbar from './components/Navbar.vue'
import Chat from './components/Chat.vue'
import About from './components/About.vue'
import EditCard from './components/EditCard.vue'

import StickerMap from './components/StickerMap.vue'

import { useI18n } from 'vue-i18n';
const i18n = useI18n();

// ------------------
import { storeToRefs } from 'pinia'
import { useChatStore } from './stores/ChatStore'

const chatStore = useChatStore()
chatStore.clear() // Clear messages on app load to ensure a clean state
const { allMessages } = storeToRefs(chatStore)
console.log('Initial messages:', allMessages.value)

// ------------------


const squareMap = "/img/map/stickers_square.png";
const rectMap = "/img/map/stickers_rect.png";
const squareCards = "/data/stickers_square.json";
const rectCards = "/data/stickers_rect.json";
const mapImage = ref(squareMap);

// define the types for the rectangles
interface Rectangle {
  id: number;
  name: string;
  bbox: [[number, number], [number, number]];
  identifier: string | undefined; // Add identifier property
  description: string | undefined; // Add description property
}


const currentRoute = ref("home");
const targets = ref<Rectangle[]>([]);
const useSquare = ref(true);
const translate = ref(false);

// ---------------------------------

const loadRectangles = async (jsonPath: string) => {
  try {
    const response = await fetch(jsonPath);
    if (!response.ok) {
      throw new Error(`Failed to load rectangles from ${jsonPath}: ${response.statusText}`);
    }
    const data = await response.json();
    const rectangles: Rectangle[] = data.map((rect: any, index: number) => ({
      id: index,
      name: `/img/card/${rect.name}` || `Rectangle ${index}`,
      bbox: rect.bbox,
    }));
    console.log(`Loaded ${rectangles.length} rectangles from ${jsonPath}`);
    targets.value = rectangles;
  } catch (error) {
    console.error('Error loading rectangles:', error);
  }
};

const currentCard = ref<string | null>(null);
const targetIdx = ref<number>(-1);

const zoomRequested = (index: number) => {
  console.log('Zoom requested for rectangle:', index);
  targetIdx.value = index;
  chatStore.setId(index); // Update the index in the chat store
  const target = targets.value[index];
  console.log('Target details:', target);
  if (!target || !target.name) {
    console.error('No target found for index:', index);
    return;
  }
  currentCard.value = target.name;
};

const restoreZoom = async () => {
  const index = chatStore.getId;
  if (index === -1) {
    console.log('No previous zoom to restore');
    return;
  }
  clearZoom(); // Clear current zoom before restoring
  await nextTick(); // Wait for the DOM to update after clearing zoom
  zoomRequested(index);
};


const clearZoom = () => {
  console.log('Clearing zoom, hiding card');
  currentCard.value = null;
  targetIdx.value = -1;
  chatStore.setId(-1); // Clear the index in the chat store as well
}


const saveZoom = () => {
  console.log('Saving zoom, hiding card');
  currentCard.value = null;
  chatStore.setId(targetIdx.value);
}

const zoomNext = () => {
  if (targetIdx.value < targets.value.length - 1) {
    zoomRequested(targetIdx.value + 1);
    chatStore.setId(targetIdx.value + 1); // Update the index in the chat store  
  }
};

const zoomPrev = () => {
  if (targetIdx.value > 0) {
    zoomRequested(targetIdx.value - 1);
    chatStore.setId(targetIdx.value - 1); // Update the index in the chat store
  }
};

onMounted(async () => {
  // remove token on app load to ensure a clean state, user needs to log in again if they want to edit cards
  localStorage.removeItem('token');
  
  const windowWidth = window.innerWidth;
  const windowHeight = window.innerHeight;
  useSquare.value = windowWidth <= windowHeight;
  try {
    if (useSquare.value) {
      await loadRectangles(squareCards);
      mapImage.value = squareMap;
    } else {
      await loadRectangles(rectCards);
      mapImage.value = rectMap;
    }
  } catch (error) {
    console.error('Error loading rectangles:', error);
  }
  // try to load descriptions from server next
  try {
    const r = await fetch(`/php/dbApi.php`, {
      headers: {
        'Content-Type': 'application/json',
      },
    })
    if (r.status === 200) {
      const cards = await r.json();
      console.log(cards);
      // match cards and rectangley on trailing name part
      targets.value = targets.value.map((rect) => {
        const namePart = rect?.name.split('/').pop();
        const card = cards.find((c: any) => c.img === namePart);
        if (card && rect) {
          console.log(`Found match for rectangle ${namePart}:`, card);
          return { ...rect, identifier: card.name, description: card.description };
        }
        return rect;
      });
    } else {
      console.error('Failed to load card descriptions from server:', r.statusText);
    }
  } catch (error) {
    console.error('Error fetching card descriptions:', error);
  }

});

const goto = (route: string) => {
  console.log('Routing to:', route);
  currentRoute.value = route;
  if (route === "home") {
    restoreZoom();
  } else if (route === "chat") {
    saveZoom();
  } else if (route === "about") {
    restoreZoom();
    console.log('Showing about section');
  } else if (route === "edit") {
    restoreZoom();
    console.log('Showing edit section');
  }
};

const toggleTx = (enabled: boolean) => {
  console.log('Toggling language, English enabled:', enabled);
  translate.value = enabled;
  if (enabled) {
    // Logic to switch to English
    console.log('Switching language to English');
    i18n.locale.value = 'en'
  } else {
    // Logic to switch to German
    console.log('Switching language to German');
    i18n.locale.value = 'de'
  }
  // Update the html lang attribute to reflect the current language
  document.documentElement.lang = i18n.locale.value;
};


</script>

<template>
  <div class="app">
    <Navbar @toggleTx="toggleTx" @route="goto" />
    <Header :route="currentRoute" />
    <main id="main-content">
      <Chat v-if="currentRoute === 'chat'" />
      <EditCard v-if="currentRoute === 'edit'" :cards="targets.map(card => card.name)" />
      <About v-if="currentRoute === 'about'" />
      <div v-if="currentRoute === 'home'" class="stickerFrame">
        <StickerMap :mapImage="mapImage" :cardImage="currentCard" :rectangles="targets" @open="zoomRequested"
          @close="clearZoom" @next="zoomNext" @prev="zoomPrev" :isSquare="useSquare" class="stickerMap" />
      </div>
      <div v-if="targetIdx !== -1" class="card-info" aria-live="polite" aria-atomic="true">
        <span v-if="targets[targetIdx]?.identifier">{{ targets[targetIdx]?.identifier }}:&nbsp;</span>
        <span v-if="targets[targetIdx]?.description">{{ targets[targetIdx]?.description }}</span>
      </div>
    </main>
    <Footer />
  </div>
</template>

<style scoped>
.stickerFrame {
  width: auto;
  margin-left: auto;
  margin-right: auto;
  overflow: hidden;
}

.stickerMap {
  width: 100%;
  height: auto;
}
</style>
