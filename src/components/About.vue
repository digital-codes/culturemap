<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import { useI18n } from 'vue-i18n';
const { t } = useI18n()

import { JsonForms } from '@jsonforms/vue';
import { RuleEffect } from '@jsonforms/core';
import { vanillaRenderers } from '@jsonforms/vue-vanilla';
import "@jsonforms/vue-vanilla/vanilla.css"


const renderers = Object.freeze([
    ...vanillaRenderers,
    // here you can add custom renderers
])

const props = defineProps({
    cards: {
        type: Array as () => string[],
        required: true
    }
})

const cardIdx = ref(0);

const cardImage = computed(() => {
    return props.cards[cardIdx.value] || "/img/card/2_5g_ovl.png";
})

const loggedIn = ref(false);
const token = ref("");

onMounted(async () => {
    console.log("About component mounted with cards: ", props.cards);
    if (props.cards.length > 0) {
        data_item.value.img = props.cards[0]?.split('/').pop() || "undefined";
        try {
            const r = await fetch(`/php/dbApi.php/?img=${data_item.value.img}`, {
                headers: {
                    'Content-Type': 'application/json',
                },
            })
            if (r.status === 200) {
                const params = await r.json();
                console.log(params);
                return;
            }
            console.log("Not found");
        }
        catch (error) {
            console.error('Error getting image:', error);
        }
    }
})

const data_usr = ref({
    validated: false,
    passkey: "",
    username: ""
})
const schema_usr = {
    "type": "object",
    "properties": {
        "username": {
            "type": "string"
        },
        "passkey": {
            "type": "string"
        }
    }
}

const uischema_usr = {
    "type": "HorizontalLayout",
    "rule": {
        "effect": RuleEffect.DISABLE,
        "condition": {
            "scope": "#/properties/validated",
            "schema": {
                "const": true
            }
        }
    },
    "elements": [
        {
            "type": "Control",
            "label": t('message.username'),
            "scope": "#/properties/username"
        },
        {
            "type": "Control",
            "label": t('message.passkey'),
            "scope": "#/properties/passkey"
        },
    ],
}

// -----------------------------

const data_item = ref({
    id : -1,
    name: "",
    url: "",
    location: "",
    geo_lat: -1000,
    geo_lng: -1000,
    img: "",
    description: "",
    tags: ""
})

const schema_item = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string"
        },
        "url": {
            "type": "string"
        },
        "location": {
            "type": "string"
        },
        "geo_lat": {
            "type": "number"
        },
        "geo_lng": {
            "type": "number"
        },
        "image": {
            "type": "string"
        },
        "description": {
            "type": "string"
        }
    }
}


const uischema_item = {
"type": "HorizontalLayout",
"elements": [
    {
        "type": "Control",
        "label": t('cardedit.name'),
        "scope": "#/properties/name"
    },
    {
        "type": "Control",
        "label": t('cardedit.description'),
        "scope": "#/properties/description"
    },
    {
        "type": "Control",
        "label": t('cardedit.url'),
        "scope": "#/properties/url"
    },
    {
        "type": "Control",
        "label": t('cardedit.location'),
        "scope": "#/properties/location"
    }
    ]
}

// -----------------------------

const data_cat = ref({
    cat_music: false,
    cat_painting: false,
    cat_media_art: false,
    cat_sports: false,
    cat_education: false,
    cat_science: false,
    cat_activism: false,
    cat_cooking: false,
    cat_crafts: false,
    cat_dancing: false,
    cat_gaming: false,
    cat_writing: false,
    cat_general_interest: false
})


const schema_cat = {
    "type": "object",
    "properties": {
        "cat_music": {
            "type": "boolean"
        },
        "cat_painting": {
            "type": "boolean"
        },
        "cat_media_art": {
            "type": "boolean"
        },
        "cat_sports": {
            "type": "boolean"
        },
        "cat_education": {
            "type": "boolean"
        },
        "cat_science": {
            "type": "boolean"
        },
        "cat_activism": {
            "type": "boolean"
        },
        "cat_cooking": {
            "type": "boolean"
        },
        "cat_crafts": {
            "type": "boolean"
        },
        "cat_dancing": {
            "type": "boolean"
        },
        "cat_gaming": {
            "type": "boolean"
        },
        "cat_writing": {
            "type": "boolean"
        },
        "cat_general_interest": {
            "type": "boolean"
        }
    }
}


const uischema_cat = {
    "type": "HorizontalLayout",
    "elements": [
        {
            "type": "Control",
            "label": t('cardedit.music'),
            "scope": "#/properties/cat_music"
        },
        {
            "type": "Control",
            "label": t('cardedit.painting'),
            "scope": "#/properties/cat_painting"
        },
        {
            "type": "Control",
            "label": t('cardedit.media_art'),
            "scope": "#/properties/cat_media_art"
        },
        {
            "type": "Control",
            "label": t('cardedit.sports'),
            "scope": "#/properties/cat_sports"
        },
        {
            "type": "Control",
            "label": t('cardedit.education'),
            "scope": "#/properties/cat_education"
        },
        {
            "type": "Control",
            "label": t('cardedit.science'),
            "scope": "#/properties/cat_science"
        },
        {
            "type": "Control",
            "label": t('cardedit.activism'),
            "scope": "#/properties/cat_activism"
        },
        {
            "type": "Control",
            "label": t('cardedit.cooking'),
            "scope": "#/properties/cat_cooking"
        },
        {
            "type": "Control",
            "label": t('cardedit.crafts'),
            "scope": "#/properties/cat_crafts"
        },
        {
            "type": "Control",
            "label": t('cardedit.dancing'),
            "scope": "#/properties/cat_dancing"
        },
        {
            "type": "Control",
            "label": t('cardedit.gaming'),
            "scope": "#/properties/cat_gaming"
        },
        {
            "type": "Control",
            "label": t('cardedit.writing'),
            "scope": "#/properties/cat_writing"
        },
        {
            "type": "Control",
            "label": t('cardedit.general_interest'),
            "scope": "#/properties/cat_general_interest"
        }
    ]
}

// -----------------------------

const onChangeUsr = async (event:any) => {
    console.log("Change event: ", event);
    data_usr.value = event.data;
    console.log("New data: ", data_usr.value);
    if (!loggedIn.value) {
        console.log("Need loging first, ignoring changes.");
        token.value = "";
        const passkey = data_usr.value.passkey;
        if (passkey === "") {
            return; // ignore
        }
        try {
            const r = await fetch('/php/llamaLogin.php', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ password: passkey, username: "any" })
            })
            if (r.status === 200) {
                const params = await r.json();
                token.value = params.token;
                loggedIn.value = true;
                data_usr.value.validated = true;
                data_usr.value.username = "";
                data_usr.value.passkey = "";
                data_item.value.img = props.cards[0]?.split('/').pop() || "undefined";
                fetchCard(data_item.value.img);
                return;
            }
            console.error('Error checking passkey:', r.statusText);
            alert("An error occurred while checking the passkey. Please try again later.");
            token.value = "";
            loggedIn.value = false;
            data_usr.value.validated = false;
            return;
        }
        catch (error) {
            console.error('Error checking passkey:', error);
            alert("An error occurred while checking the passkey. Please try again later.");
            return;
        }
    }
}

const onChangeItem = async (event:any) => {
    console.log("Change event: ", event);
    data_item.value = event.data;
    console.log("New data: ", data_item.value);
}
const onChangeCat = async (event:any) => {
    console.log("Change event: ", event);
    data_cat.value = event.data;
    console.log("New data: ", data_cat.value);
}

const submit = async () => {
    console.log("Submitting data: ", data_item.value, data_cat.value);
    try {
        data_item.value.tags = "";
        for (const [key, value] of Object.entries(data_cat.value)) {
            if (value) {
                data_item.value.tags += key.replace("cat_", "") + ",";
            }
        }
        data_item.value.tags = data_item.value.tags.replace(/,+$/, ''); // strip trailing ,
        if (data_item.value.id === -1) {
            const r = await fetch('/php/dbApi.php', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token.value}`
                },
                body: JSON.stringify({ "token": token.value, "name": data_item.value.name, "url": data_item.value.url, "location": data_item.value.location, "geo_lat": data_item.value.geo_lat, "geo_lng": data_item.value.geo_lng, "description": data_item.value.description, "tags": data_item.value.tags, "img": data_item.value.img })
            })
            if (r.status === 200) {
                alert("Group submitted successfully!");
                return;
            }
            console.error('Error submitting group:', r.statusText);
            alert("An error occurred while submitting the group. Please try again later.");
            return;
        } else {
            const r = await fetch('/php/dbApi.php', {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token.value}`                },
                body: JSON.stringify({ "token": token.value, "id": data_item.value.id, "name": data_item.value.name, "url": data_item.value.url, "location": data_item.value.location, 
                "geo_lat": data_item.value.geo_lat, "geo_lng": data_item.value.geo_lng, "description": data_item.value.description, "tags": data_item.value.tags, "img": data_item.value.img })
            })
            if (r.status === 200) {
                alert("Group updated successfully!");
                return;
            }
            console.error('Error updating group:', r.statusText);
            alert("An error occurred while updating the group. Please try again later.");
            return;
        }
    }
    catch (error) {
        console.error('Error submitting group:', error);
        alert("An error occurred while submitting the group. Please try again later.");
        return;
    }
}

const clearItem = () => {
    data_item.value.id = -1;
    data_item.value.name = "";
    data_item.value.url = "";
    data_item.value.location = "";
    data_item.value.geo_lat = -1000;
    data_item.value.geo_lng = -1000;
    data_item.value.description = "";
    data_cat.value.cat_music = false;
    data_cat.value.cat_painting = false;
    data_cat.value.cat_media_art = false;
    data_cat.value.cat_sports = false;
    data_cat.value.cat_education = false;
    data_cat.value.cat_science = false;
    data_cat.value.cat_activism = false;
    data_cat.value.cat_cooking = false;
    data_cat.value.cat_crafts = false;
    data_cat.value.cat_dancing = false;
    data_cat.value.cat_gaming = false;
    data_cat.value.cat_writing = false;
    data_cat.value.cat_general_interest = false;
}

const previous = async () => {
    if (cardIdx.value > 0) {
        cardIdx.value -= 1;
        clearItem();
        data_item.value.img = props.cards[cardIdx.value]?.split('/').pop() || "undefined";
        fetchCard(data_item.value.img);
        console.log("Previous card",data_item.value.img);
    }
}

const next = async () => {    
    if (cardIdx.value < props.cards.length - 1) {
        cardIdx.value += 1;
        clearItem();
        data_item.value.img = props.cards[cardIdx.value]?.split('/').pop() || "undefined";
        fetchCard(data_item.value.img);
        console.log("Next card",data_item.value.img);
    }
}

const fetchCard = async (cardName: string) => {
    try {
        const r = await fetch(`/php/dbApi.php/?img=${cardName}`, {
            headers: {
                'Content-Type': 'application/json',
            },
        })
        if (r.status === 200) {
            const params = await r.json();
            console.log(params);
            if (params && params.id) {
                data_item.value.id = parseInt(params.id);
            }
            if (params && params.name) {
                data_item.value.name = params.name;
            }
            if (params && params.url) {
                data_item.value.url = params.url;
            }
            if (params && params.location) {
                data_item.value.location = params.location;
            }
            if (params && params.geo_lat) {
                data_item.value.geo_lat = parseFloat(params.geo_lat);
            }
            if (params && params.geo_lng) {
                data_item.value.geo_lng = parseFloat(params.geo_lng);
            }
            if (params && params.description) {
                data_item.value.description = params.description;
            }
            if (params && params.tags) {
                const tagsArray = params.tags.split(',').map((tag: string) => tag.trim());
                data_cat.value.cat_music = tagsArray.includes("music");
                data_cat.value.cat_painting = tagsArray.includes("painting");
                data_cat.value.cat_media_art = tagsArray.includes("media_art");
                data_cat.value.cat_sports = tagsArray.includes("sports");
                data_cat.value.cat_education = tagsArray.includes("education");
                data_cat.value.cat_science = tagsArray.includes("science");
                data_cat.value.cat_activism = tagsArray.includes("activism");
                data_cat.value.cat_cooking = tagsArray.includes("cooking");
                data_cat.value.cat_crafts = tagsArray.includes("crafts");
                data_cat.value.cat_dancing = tagsArray.includes("dancing");
                data_cat.value.cat_gaming = tagsArray.includes("gaming");
                data_cat.value.cat_writing = tagsArray.includes("writing");
                data_cat.value.cat_general_interest = tagsArray.includes("general_interest");
            }
            return;
        }
        console.log("Not found");
    }
    catch (error) {
        console.error('Error getting image:', error);
    }
}

</script>

<template>
    <div class="about">
        <!-- 
        <p>{{ $t('message.about') }}</p>
        -->
        <div class="cardview" v-if="loggedIn">
            <button class="slide" @click="previous"><span class="icon">arrow_back</span></button>
            <img :src="cardImage" alt="Group Image" class="cardimage" />
            <button class="slide" @click="next"><span class="icon">arrow_forward</span></button>
        </div>
        <div>
            <button v-if="loggedIn" class="submit" @click="submit">{{ $t('message.send') }}</button>
        </div>
        <json-forms class="login" :class="loggedIn? 'login_done' : 'login_pending'" :data="data_usr" :schema="schema_usr" :uischema="uischema_usr" :renderers="renderers"
            @change="onChangeUsr" />
        <json-forms v-if="data_usr.validated" class="items" :data="data_item" :schema="schema_item" :uischema="uischema_item" :renderers="renderers" @change="onChangeItem" />
        <json-forms v-if="data_usr.validated" class="categories" :data="data_cat" :schema="schema_cat" :uischema="uischema_cat" :renderers="renderers"
            @change="onChangeCat" />

    </div>
</template>

<style scoped>
.cardview {
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 1rem;
}

.cardimage {
    max-width: 240px;
    max-height: 240px;
    margin-bottom: .5rem;
}

.slide {
    color: var(--color-text);
    background-color: var(--color-background);
    border: none;
    font-size: 24px;
    cursor: pointer;
}

.login_pending {
    border-style: solid;
    border-color: red;
    border-width: 1px;
    padding: .5rem;
}
.login_done {
    border-style: solid;
    border-color: green;
    border-width: 1px;
    padding: .5rem;
}

.items {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: .2rem;
    margin-top: .1rem;
}

.categories {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: .2rem;
    margin-top: .1rem;
}

.submit {
    background-color: #4CAF50; /* Green */
    border: none;
    color: white;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin-bottom: 1rem;
    cursor: pointer;
}

</style>

<style> 
@media (max-width: 600px) {
    .login .input {
        align-items: center;
        max-width: 70%;
    }
    .items {
        grid-template-columns: 1fr;
    }
    .categories {
        grid-template-columns: 1fr;
    }
}
.horizontal-layout-item .control .description {
    display:none;
}

</style>
