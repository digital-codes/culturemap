<script setup lang="ts">
import { ref, onMounted } from 'vue';

import Header from './components/Header.vue'
import Footer from './components/Footer.vue'
import Navbar from './components/Navbar.vue'
import Chat from './components/Chat.vue'

import StickerMap from './components/StickerMap.vue'

import { useI18n } from 'vue-i18n';
const i18n = useI18n();

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
}



const targets = ref<Rectangle[]>([]);
const useSquare = ref(true);

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
  const target = targets.value[index];
  if (!target || !target.name) {
    console.error('No target found for index:', index);
    return;
  }
  currentCard.value = target.name;
};


const clearZoom = () => {
  console.log('Clearing zoom, hiding card');
  currentCard.value = null;
  targetIdx.value = -1;
}

const zoomNext = () => {
  if (targetIdx.value < targets.value.length - 1) {
    zoomRequested(targetIdx.value + 1);
  }
};

const zoomPrev = () => {
  if (targetIdx.value > 0) {
    zoomRequested(targetIdx.value - 1);
  }
};

onMounted(() => {
  const windowWidth = window.innerWidth;
  const windowHeight = window.innerHeight;
  useSquare.value = windowWidth <= windowHeight;
  try {
    if (useSquare.value) {
      loadRectangles(squareCards);
      mapImage.value = squareMap;
    } else {
      loadRectangles(rectCards);
      mapImage.value = rectMap;
    }
  } catch (error) {
    console.error('Error loading rectangles:', error);
  }
});

const chatEnabled = ref(false);

const toggleChat = (enabled: boolean) => {
  console.log('Toggling chat, enabled:', enabled);

  chatEnabled.value = enabled;

  if (enabled) {
    // Logic to show the chat component
    console.log('Showing chat component');
  } else {
    // Logic to hide the chat component
    console.log('Hiding chat component');
  }
  // Implement any additional logic needed when chat is toggled
};

const translate = ref(false);

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
  // Implement logic to switch language based on the value of 'enabled'
  // For example, you could set a reactive variable that controls the language of the app
};


</script>

<template>
  <div class="app">
    <Navbar @toggleChat="toggleChat" @toggleTx="toggleTx" />
    <Header />
    <Chat v-if="chatEnabled" />
    <div v-else class="stickerFrame">
      <StickerMap :mapImage="mapImage" :cardImage="currentCard" :rectangles="targets" @open="zoomRequested"
        @close="clearZoom" 
        @next="zoomNext" @prev="zoomPrev"
        :isSquare="useSquare" class="stickerMap" />
    </div>
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
