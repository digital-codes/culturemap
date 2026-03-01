<script setup lang="ts">
import { ref, onMounted } from 'vue';
import StickerMap from './components/StickerMap.vue'


const squareMap = "/img/map/stickers_square.png";
const rectMap = "/img/map/stickers_rect.png";
const squareCards = "/data/stickers_square.json";
const rectCards = "/data/stickers_rect.json";
const mapImage = ref(squareMap);

// define the types for the rectangles
interface Rectangle {
  id: number;
  name: string;
  bbox: [number, number, number, number]; // [x, y, width, height]
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
      name: rect.path || `Rectangle ${index}`,
      bbox: rect.bbox,
    }));
    console.log(`Loaded ${rectangles.length} rectangles from ${jsonPath}`);
    targets.value = rectangles;
  } catch (error) {
    console.error('Error loading rectangles:', error);
  }
};

const zoomRequested = (index: number) => {
  console.log('Zoom requested for rectangle:', index);
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

</script>

<template>
  <StickerMap :mapImage="mapImage" :rectangles="targets" @rectangle-clicked="zoomRequested"/>
</template>

<style scoped>
.logo {
  height: 6em;
  padding: 1.5em;
  will-change: filter;
  transition: filter 300ms;
}
.logo:hover {
  filter: drop-shadow(0 0 2em #646cffaa);
}
.logo.vue:hover {
  filter: drop-shadow(0 0 2em #42b883aa);
}
</style>
